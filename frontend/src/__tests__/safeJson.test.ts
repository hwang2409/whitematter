import { describe, it, expect } from "vitest";
import { safeParseJson } from "@/lib/safeJson";

describe("safeParseJson", () => {
  it("parses valid JSON response", async () => {
    const data = { name: "test", value: 42 };
    const res = new Response(JSON.stringify(data), { status: 200 });
    const result = await safeParseJson(res);
    expect(result).toEqual(data);
  });

  it("parses valid JSON array response", async () => {
    const data = [1, 2, 3];
    const res = new Response(JSON.stringify(data), { status: 200 });
    const result = await safeParseJson(res);
    expect(result).toEqual(data);
  });

  it("throws on HTML response (starts with <)", async () => {
    const html = "<html><body>Not Found</body></html>";
    const res = new Response(html, { status: 404 });
    await expect(safeParseJson(res)).rejects.toThrow(
      "Server returned HTML instead of JSON"
    );
  });

  it("throws on HTML response with leading whitespace", async () => {
    const html = "  <html><body>Error</body></html>";
    const res = new Response(html, { status: 500 });
    await expect(safeParseJson(res)).rejects.toThrow(
      "Server returned HTML instead of JSON"
    );
  });

  it("returns {} on empty response", async () => {
    const res = new Response("", { status: 200 });
    const result = await safeParseJson(res);
    expect(result).toEqual({});
  });

  it("returns {} on whitespace-only response", async () => {
    const res = new Response("   \n  ", { status: 200 });
    const result = await safeParseJson(res);
    expect(result).toEqual({});
  });

  it("throws on invalid JSON", async () => {
    const res = new Response("not valid json {{{", { status: 200 });
    await expect(safeParseJson(res)).rejects.toThrow(
      "Invalid JSON response from server."
    );
  });
});
