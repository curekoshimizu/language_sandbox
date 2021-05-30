import Box from '@material-ui/core/Box';

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
    </Box>
  );
}) as React.FC;
