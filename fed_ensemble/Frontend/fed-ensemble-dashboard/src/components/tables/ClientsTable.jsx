/* eslint-disable react/prop-types */
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Chip } from '@mui/material';
import { computeMean, getAnomalyChipColor, getAnomalyStatus, getStatusColor } from '../../utils/dataHelpers';

const ClientsTable = ({ clients, anomalyThreshold }) => {
  return (
    <TableContainer component={Paper} elevation={0} variant="outlined">
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Client ID</TableCell>
            <TableCell align='center'>Average Accuracy</TableCell>
            <TableCell align='center'>Average Loss</TableCell>
            <TableCell align='center'>Anomaly Score <br/> (Average Round)</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {clients
            ?.sort((a, b) => a.node_id - b.node_id)
            .map((client) => (
              <TableRow key={client.node_id}>
                <TableCell>Client {client.node_id}</TableCell>
                <TableCell align='center'>{(((computeMean(client.accuracy))) * 100).toFixed(2)}%</TableCell>
                <TableCell align='center'>{computeMean(client.loss).toFixed(2)}</TableCell>
                <TableCell align='center'>
                  <Chip 
                    label={parseFloat(computeMean(client.anomaly_scores)).toFixed(2)} 
                    color={getAnomalyChipColor(computeMean(client.anomaly_scores), anomalyThreshold)}
                    size="small"
                  />
                </TableCell>
                <TableCell
                  sx={{
                    color: getStatusColor(getAnomalyStatus(computeMean(client.anomaly_scores), anomalyThreshold))
                  }}
                >
                  {getAnomalyStatus(computeMean(client.anomaly_scores), anomalyThreshold)}
                </TableCell>
              </TableRow>
            ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default ClientsTable; 