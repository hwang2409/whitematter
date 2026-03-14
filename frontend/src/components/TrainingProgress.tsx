import * as api from "@/api";
import TrainingChart from "./TrainingChart";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Paper from "@mui/material/Paper";
import LinearProgress from "@mui/material/LinearProgress";

interface Props {
  trainingJob: api.CustomTrainJob;
  trainingHistory: { epoch: number; loss: number; accuracy: number }[];
  currentEpoch: number;
  totalEpochs: number;
}

export default function TrainingProgress({ trainingJob, trainingHistory, currentEpoch, totalEpochs }: Props) {
  return (
    <Paper variant="outlined" sx={{ mt: 2, p: 2, borderColor: "divider" }}>
      <Typography variant="h3" sx={{ mb: 1.5 }}>
        Training Progress
      </Typography>
      <Box sx={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 1, mb: 1.25 }}>
        <Box sx={{ textAlign: "center", p: 1, bgcolor: "background.default", borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", display: "block", mb: 0.5 }}>
            Status
          </Typography>
          <Typography variant="body1" fontWeight={600} sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
            {trainingJob.status}
          </Typography>
        </Box>
        <Box sx={{ textAlign: "center", p: 1, bgcolor: "background.default", borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", display: "block", mb: 0.5 }}>
            Epoch
          </Typography>
          <Typography variant="body1" fontWeight={600} sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
            {currentEpoch} / {totalEpochs}
          </Typography>
        </Box>
        {"loss" in trainingJob && (
          <Box sx={{ textAlign: "center", p: 1, bgcolor: "background.default", borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", display: "block", mb: 0.5 }}>
              Loss
            </Typography>
            <Typography variant="body1" fontWeight={600} sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
              {(trainingJob.loss || 0).toFixed(4)}
            </Typography>
          </Box>
        )}
        {"accuracy" in trainingJob && (
          <Box sx={{ textAlign: "center", p: 1, bgcolor: "background.default", borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", display: "block", mb: 0.5 }}>
              Accuracy
            </Typography>
            <Typography variant="body1" fontWeight={600} sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
              {(trainingJob.accuracy || 0).toFixed(2)}%
            </Typography>
          </Box>
        )}
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
        {trainingJob.message}
      </Typography>
      {["training", "running"].includes(trainingJob.status) && currentEpoch > 0 && (
        <LinearProgress
          variant="determinate"
          value={(currentEpoch / totalEpochs) * 100}
          sx={{ mt: 1, height: 3, borderRadius: 1, "& .MuiLinearProgress-bar": { borderRadius: 1 } }}
        />
      )}
      {trainingHistory.length > 0 && <TrainingChart data={trainingHistory} />}
      {trainingJob.status === "completed" && (
        <Box
          sx={{
            mt: 1.5,
            p: 1.25,
            border: "1px solid",
            borderColor: "success.main",
            borderRadius: 1,
          }}
        >
          <Typography variant="body2" sx={{ mb: 0.5 }}>
            Training completed! Model ID: <Box component="code" sx={{ fontFamily: '"JetBrains Mono", monospace', bgcolor: "action.hover", px: 0.5, borderRadius: 0.5 }}>{trainingJob.model_id}</Box>
          </Typography>
          <Typography variant="caption" color="text.secondary">
            API endpoint: <Box component="code" sx={{ fontFamily: '"JetBrains Mono", monospace' }}>POST /api/{trainingJob.model_id}/predict</Box>
          </Typography>
        </Box>
      )}
    </Paper>
  );
}
