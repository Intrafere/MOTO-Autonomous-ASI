import React from 'react';
import { autonomousAPI } from '../../services/api';
import ApiCallLogs from '../ApiCallLogs';
import '../autonomous/AutonomousResearch.css';

export default function LeanOJLogs() {
  return (
    <div className="autonomous-logs">
      <div className="autonomous-header">
        <h2>API Call Logs</h2>
      </div>

      <ApiCallLogs
        api={autonomousAPI}
        workflow="leanoj"
        emptyHint="Run Proof Solver and make model calls to see request/response logs here."
      />
    </div>
  );
}
