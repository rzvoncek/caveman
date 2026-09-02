package gateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/JuliusBrussee/caveman/proxy/providers"
	"github.com/JuliusBrussee/caveman/proxy/providers/openai"
)

// newBobTestServer builds a minimal server wired to a Bob upstream stub and a
// specific runtime mode, using the shared test seams.
func newBobTestServer(t *testing.T, bobUpstreamURL string, rc RequestContext) *Server {
	t.Helper()
	return New(Config{
		Adapters:    []providers.Adapter{openai.New(bobUpstreamURL)},
		Auth:        stubAuth{rc: rc},
		Creds:       stubCreds{key: "token-unused-bob-passes-inbound-auth"},
		BobUpstream: bobUpstreamURL,
		HTTPClient:  &http.Client{},
	})
}

// TestBobAdminPathPassThrough proves the /bob/ route forwards /admin/* requests
// byte-identically to the IBM gateway. This is the unblocking path: Bob's startup
// GET /admin/v1/profile must succeed or it triggers an SSO crash.
func TestBobAdminPathPassThrough(t *testing.T) {
	const respBody = `{"status":"ok","profile":"test"}`
	var gotPath string
	var gotAuthHeader string

	ibmGateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuthHeader = r.Header.Get("authorization")
		w.Header().Set("content-type", "application/json")
		io.WriteString(w, respBody)
	}))
	defer ibmGateway.Close()

	srv := newBobTestServer(t, ibmGateway.URL, RequestContext{Label: "local", RuntimeMode: "compress"})

	req := httptest.NewRequest(http.MethodGet, "/bob/admin/v1/profile", nil)
	req.Header.Set("authorization", "Bearer bob-sso-token")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("admin path: status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	if rec.Body.String() != respBody {
		t.Errorf("admin path: response not byte-identical:\n got %q\nwant %q", rec.Body.String(), respBody)
	}
	if gotPath != "/admin/v1/profile" {
		t.Errorf("admin path: upstream received path %q, want /admin/v1/profile", gotPath)
	}
	if gotAuthHeader != "Bearer bob-sso-token" {
		t.Errorf("admin path: Authorization header not forwarded: got %q", gotAuthHeader)
	}
}

// TestBobInferencePathForwarded proves that POST /inference/v1/chat/completions
// reaches the upstream with the request body intact (no compression attempted
// in record mode, and no panic in compress mode with nil compressor).
func TestBobInferencePathForwarded(t *testing.T) {
	const reqBody = `{"model":"ibm/granite-3-3-8b-instruct","messages":[{"role":"user","content":"hi"}]}`
	const respBody = `{"id":"chatcmpl-stub","choices":[{"message":{"role":"assistant","content":"hello"}}]}`
	var gotPath string
	var gotBody string

	ibmGateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("content-type", "application/json")
		io.WriteString(w, respBody)
	}))
	defer ibmGateway.Close()

	// compress mode with nil compressor: must fall through to pass-through
	srv := newBobTestServer(t, ibmGateway.URL, RequestContext{Label: "local", RuntimeMode: "compress"})

	req := httptest.NewRequest(http.MethodPost, "/bob/inference/v1/chat/completions",
		strings.NewReader(reqBody))
	req.Header.Set("authorization", "Bearer bob-api-key")
	req.Header.Set("content-type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("inference path: status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	if rec.Body.String() != respBody {
		t.Errorf("inference path: response not byte-identical:\n got %q\nwant %q", rec.Body.String(), respBody)
	}
	if gotPath != "/inference/v1/chat/completions" {
		t.Errorf("inference path: upstream received path %q, want /inference/v1/chat/completions", gotPath)
	}
	if gotBody != reqBody {
		t.Errorf("inference path: request body not forwarded:\n got %q\nwant %q", gotBody, reqBody)
	}
}

// TestBobRoutePreservesQueryString proves query parameters are forwarded intact.
func TestBobRoutePreservesQueryString(t *testing.T) {
	var gotQuery string
	ibmGateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ibmGateway.Close()

	srv := newBobTestServer(t, ibmGateway.URL, RequestContext{Label: "local", RuntimeMode: "record"})

	req := httptest.NewRequest(http.MethodGet, "/bob/admin/v1/health?foo=bar&baz=1", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("query: status = %d, want 204", rec.Code)
	}
	if gotQuery != "foo=bar&baz=1" {
		t.Errorf("query: upstream received query %q, want foo=bar&baz=1", gotQuery)
	}
}
