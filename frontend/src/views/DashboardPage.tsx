"use client";
import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import "./DashboardPage.css";

export default function DashboardPage() {
  const { user } = useAuth();

  return (
    <div className="dashboard-page">
      <h2>Dashboard</h2>
      <p className="dashboard-welcome">Welcome, {user?.email}</p>
      <div className="dashboard-links">
        <Link href="/data">Data (S3)</Link>
        <Link href="/train">Train</Link>
        <Link href="/models">Models</Link>
        <Link href="/settings">AWS settings</Link>
      </div>
      <p className="dashboard-note">
        BYOC training jobs and model architectures can be listed here once the API is wired.
      </p>
    </div>
  );
}
