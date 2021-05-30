import Box from '@material-ui/core/Box';
import Typography from '@material-ui/core/Typography';

const Alignment = () => {
  return (
    <>
      <Typography component="h2" gutterBottom variant="h3">
        Alignment
      </Typography>

      <Box
        alignItems="center"
        bgcolor="lightblue"
        display="flex"
        height={130}
        justifyContent="center"
        width={500}
      >
        <Box border={1} height={40} width={100}>
          Linux
        </Box>
        <Box border={1} height={40} width={100}>
          Windows
        </Box>
        <Box border={1} height={40} width={100}>
          Mac
        </Box>
      </Box>
      <Box
        alignItems="center"
        bgcolor="lightgreen"
        display="flex"
        height={80}
        justifyContent="center"
        width={500}
      >
        2. lightgreen
      </Box>
      <Box
        alignItems="center"
        bgcolor="pink"
        display="flex"
        height={80}
        justifyContent="center"
        width={500}
      >
        3. pink
      </Box>
    </>
  );
};

export default (() => {
  return (
    <Box width="100%">
      <Box bgcolor="grey.400" my={0.5} p={1} width={1 / 4}>
        Width 1/4
      </Box>
      <Box bgcolor="grey.400" my={0.5} p={1} width={400}>
        Width 400
      </Box>
      <Box bgcolor="grey.400" my={0.5} p={1} width="400px">
        Width 400px
      </Box>
      <Box bgcolor="grey.400" my={0.5} p={1} width="400pt">
        Width 400pt
      </Box>
      <Box bgcolor="grey.400" my={0.5} p={1} width="75%">
        Width 75%
      </Box>
      <Box bgcolor="grey.400" my={0.5} p={1} width={1}>
        Width 1
      </Box>
      <Alignment />
    </Box>
  );
}) as React.FC;
