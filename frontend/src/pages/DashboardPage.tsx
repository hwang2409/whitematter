import { useAuth } from "../context/AuthContext";
import "./DashboardPage.css";

export default function DashboardPage() {
  const { user } = useAuth();

  return (
    <div className="dashboard-page">
      <h2>Dashboard</h2>
      <p className="dashboard-welcome">Welcome, {user?.email}</p>
      <div className="dashboard-links">
        <a href="/#data">Data (S3)</a>
        <a href="/#train">Train</a>
        <a href="/#models">Models</a>
        <a href="/#settings">AWS settings</a>
      </div>
      <p className="dashboard-note">
        BYOC training jobs and model architectures can be listed here once the API is wired.
      </p>
    </div>
  );
}
