import { Variant } from '@material-ui/core/styles/createTypography';
import Typography from '@material-ui/core/Typography';

const range = (start: number, length: number): number[] =>
  [...Array(length).keys()].map((i) => start + i);

export default (() => {
  return (
    <>
      {range(1, 6).map((i) => {
        return (
          <Typography component="h1" variant={`h${i}` as Variant}>
            H1 tag.
            {i !== 1 ? ` but H${i} style.` : ''}
          </Typography>
        );
      })}
    </>
  );
}) as React.FC;
