package gateway

import (
	"bytes"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/JuliusBrussee/caveman/proxy/providers"
	"github.com/JuliusBrussee/caveman/proxy/providers/openai"
	"github.com/JuliusBrussee/caveman/shared/platform/httpx"
	"github.com/JuliusBrussee/caveman/shared/platform/id"
)

// DefaultBobUpstream is IBM Bob's default gateway. BOB_GATEWAY_URL points here
// by default; individual region overrides are supported via caveman.yaml or the
// CAVEMAN_BOB_UPSTREAM env var.
const DefaultBobUpstream = "https://api.us-east.bob.ibm.com"

// bobInferencePath is the inference sub-path the proxy compresses.
// All other Bob paths (e.g. /admin/v1/*) are forwarded byte-identically.
const bobInferencePath = "/inference/v1/chat/completions"

// bob is the IBM Bob gateway route. All Bob traffic arrives at /bob/<path> when
// BOB_GATEWAY_URL is pointed at the proxy. It:
//   - forwards every non-inference path (e.g. /admin/v1/*) byte-identically to
//     the IBM gateway, preserving auth headers — this is the unblocking path that
//     lets Bob's startup /admin/v1/profile call succeed
//   - runs POST /inference/v1/chat/completions through the same live-zone
//     compress path the OpenAI adapter uses, falling back to a byte-identical
//     pass-through on any error (byte-safe invariant preserved)
//
// Invariants (same as chatgpt.go):
//   - no credential resolution or substitution — the agent's own Authorization
//     header passes through untouched
//   - any parse/store/compress failure forwards the original bytes unchanged
//   - response bodies stay byte-identical; SSE streams stay unbuffered
func (s *Server) bob(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	requestID := id.NewUUIDv7()
	traceID := traceIDFrom(r)
	w.Header().Set("x-cave-request-id", requestID)
	w.Header().Set("x-cave-trace-id", traceID)

	rc, err := s.auth.Authenticate(r.Context(), r)
	if err != nil {
		httpx.Error(w, r, http.StatusUnauthorized, "cave_unauthorized", "Request rejected by the proxy authenticator.")
		return
	}
	rc.AgentSlug = labelOrDefault(r.Header.Get("x-cave-agent"), "bob")

	// Strip the /bob prefix to get the raw IBM path.
	suffix := strings.TrimPrefix(r.URL.Path, "/bob")
	if suffix == "" {
		suffix = "/"
	}
	upstreamURL := s.bobUpstream + suffix
	if r.URL.RawQuery != "" {
		upstreamURL += "?" + r.URL.RawQuery
	}

	// Only POST to the inference path is a compression candidate. Everything else
	// (admin health, profile, metrics, etc.) goes straight through.
	lockedRoutes, compiledPlanAllowed := compiledPlanRoutes(r.Header)
	evidence := requestEvidenceFromHeaders(r.Header)
	adapter := openai.New(s.bobUpstream)
	compressEligible := r.Method == http.MethodPost &&
		suffix == bobInferencePath &&
		rc.RuntimeMode == "compress" &&
		s.compressor != nil &&
		s.liveZoneCompressionAllowed(adapter, nil) &&
		compiledPlanAllowed

	const capLimit = 4 << 20
	transform := providers.TransformResult{OptimizerIDs: []string{}}
	var comp *compressionOutcome
	var originalBody []byte
	var reqBody io.Reader

	if compressEligible {
		captured, readErr := io.ReadAll(io.LimitReader(r.Body, capLimit+1))
		if readErr == nil && len(captured) <= capLimit {
			originalBody = captured
			transform.Body = originalBody
			headersForInspect := r.Header.Clone()
			headersForInspect.Set("x-cave-route-path", suffix)
			meta, inspectErr := adapter.InspectRequest(r.Context(), bytes.NewReader(originalBody), headersForInspect)
			if inspectErr == nil {
				meta.Endpoint = suffix
				meta.SessionID = evidence.SessionID
				if s.cacheEpochAllows(r, adapter, meta, originalBody, evidence.SessionID) {
					comp = s.compressRequest(adapter, originalBody, meta, &transform, requestID, lockedRoutes)
				}
			}
			reqBody = bytes.NewReader(transform.Body)
		} else {
			// Over-limit or read error: stream through unchanged.
			reqBody = io.MultiReader(bytes.NewReader(captured), r.Body)
		}
	} else {
		reqBody = r.Body
	}
	if r.ContentLength == 0 && (r.Method == http.MethodGet || r.Method == http.MethodHead || r.Method == http.MethodDelete) {
		reqBody = nil
	}

	upReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, reqBody)
	if err != nil {
		httpx.Error(w, r, http.StatusBadRequest, "cave_provider_request_invalid", "Bob route could not build the upstream request.")
		return
	}
	if transform.Body != nil {
		upReq.ContentLength = int64(len(transform.Body))
	} else {
		upReq.ContentLength = r.ContentLength
	}
	upReq.Header = chatGPTRequestHeaders(r.Header) // strips proxy-private + hop-by-hop, preserves auth

	var resp *http.Response
	if transform.Body != nil {
		resp, err = s.doUpstream(r.Context(), func() (*http.Request, error) {
			req, buildErr := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, bytes.NewReader(transform.Body))
			if buildErr != nil {
				return nil, buildErr
			}
			req.ContentLength = int64(len(transform.Body))
			req.Header = chatGPTRequestHeaders(r.Header)
			return req, nil
		})
	} else {
		resp, err = s.httpClient.Do(upReq)
	}
	if err != nil {
		httpx.Error(w, r, http.StatusBadGateway, "cave_upstream_unavailable", "IBM Bob upstream is unreachable.")
		return
	}
	// Byte-safe fail-open: if the upstream rejected a transformed request, retry
	// once with the original bytes — same pattern as proxy.go and chatgpt.go.
	if resp.StatusCode >= 400 && resp.StatusCode < 500 && comp != nil && originalBody != nil {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
		_ = resp.Body.Close()
		retryResp, doErr := s.doUpstream(r.Context(), func() (*http.Request, error) {
			retryReq, retryErr := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, bytes.NewReader(originalBody))
			if retryErr != nil {
				return nil, retryErr
			}
			retryReq.Header = chatGPTRequestHeaders(r.Header)
			return retryReq, nil
		})
		if doErr != nil {
			httpx.Error(w, r, http.StatusBadGateway, "cave_upstream_unavailable", "IBM Bob upstream is unreachable.")
			return
		}
		resp = retryResp
		comp = nil
		transform = providers.TransformResult{Body: originalBody, OptimizerIDs: []string{}}
	}
	defer resp.Body.Close()

	copySafeResponseHeaders(w.Header(), resp.Header)
	if comp != nil {
		w.Header().Set("x-cave-mode", rc.RuntimeMode)
		w.Header().Set("x-cave-optimization", strings.Join(transform.OptimizerIDs, ","))
		w.Header().Set("x-caveman-compression-ratio", strconv.FormatFloat(comp.ratio, 'f', 4, 64))
		w.Header().Set("x-caveman-recovery-handle", comp.handle)
		w.Header().Set("x-caveman-tokens-before", strconv.Itoa(comp.before))
		w.Header().Set("x-caveman-tokens-after", strconv.Itoa(comp.after))
		w.Header().Set("x-caveman-token-count-basis", "estimated_engine_o200k")
	}
	w.WriteHeader(resp.StatusCode)
	s.streamThrough(w, resp.Body)

	if s.logger != nil {
		s.logger.Info("bob_proxy",
			"path", suffix, "status", resp.StatusCode,
			"latency_ms", time.Since(start).Milliseconds(), "compressed", comp != nil)
	}
}
