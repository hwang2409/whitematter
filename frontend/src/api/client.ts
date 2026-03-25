// API base URL - empty string uses relative URLs (Next.js rewrites to backend)
// Set NEXT_PUBLIC_API_BASE environment variable to override
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

// Request timeout in milliseconds (30 seconds)
export const REQUEST_TIMEOUT = 30000;

// Error types for better error handling
export type ApiErrorType = 'timeout' | 'network' | 'server' | 'unknown';

export class ApiError extends Error {
  type: ApiErrorType;
  statusCode?: number;

  constructor(message: string, type: ApiErrorType, statusCode?: number) {
    super(message);
    this.name = 'ApiError';
    this.type = type;
    this.statusCode = statusCode;
  }
}

// Helper to create fetch with timeout
export async function fetchWithTimeout(
  url: string,
  options?: RequestInit,
  timeout: number = REQUEST_TIMEOUT
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new ApiError(
        `Server error: ${response.status} ${response.statusText}`,
        'server',
        response.status
      );
    }

    return response;
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof ApiError) {
      throw error;
    }

    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new ApiError(
          'Request timed out. The server took too long to respond.',
          'timeout'
        );
      }

      // Network errors (no internet, server unreachable, etc.)
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        throw new ApiError(
          'Network error. Please check your connection and try again.',
          'network'
        );
      }
    }

    throw new ApiError(
      error instanceof Error ? error.message : 'An unknown error occurred',
      'unknown'
    );
  }
}

// Helper to get user-friendly error message
export function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
}

// Helper to check if error is retryable
export function isRetryableError(error: unknown): boolean {
  if (error instanceof ApiError) {
    return error.type === 'timeout' || error.type === 'network';
  }
  return false;
}
