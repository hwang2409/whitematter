"use client";
import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { getMe, getStoredToken, storeTokens, clearTokens } from "@/services/auth";

interface User {
  id: string;
  email: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  loading: boolean;
  loginWithTokens: (tokens: { access_token: string; refresh_token: string; token_type: string }) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(getStoredToken());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      getMe(token)
        .then((u) => setUser({ id: u.id, email: u.email }))
        .catch(() => {
          clearTokens();
          setToken(null);
          setUser(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, [token]);

  const loginWithTokens = (tokens: { access_token: string; refresh_token: string; token_type: string }) => {
    storeTokens(tokens);
    setToken(tokens.access_token);
  };

  const logout = () => {
    clearTokens();
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, loginWithTokens, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be inside AuthProvider");
  return ctx;
}
