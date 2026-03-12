/**
 * Parse response as JSON. If the server returns HTML (e.g. 404/error page),
 * throws a clear error instead of "Unexpected token '<'" SyntaxError.
 */
export async function safeParseJson<T = unknown>(res: Response): Promise<T> {
  const text = await res.text();
  const trimmed = text.trim();
  if (trimmed.startsWith("<")) {
    throw new Error(
      "Server returned HTML instead of JSON. Start the API (e.g. in another terminal: cd platform && python server.py). It runs on port 8080; or set NEXT_PUBLIC_API_BASE to your API URL. See frontend/README.md."
    );
  }
  if (!trimmed) return {} as T;
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error("Invalid JSON response from server.");
  }
}
