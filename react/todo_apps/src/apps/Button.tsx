import { Button } from '@material-ui/core';
import { ButtonTypeMap } from '@material-ui/core/Button';

export default (() => {
  const variants: Array<ButtonTypeMap['props']['variant']> = [
    'contained',
    'outlined',
    'text',
  ];
  return (
    <>
      {variants.map((variant) => (
        <div key={variant}>
          <Button variant={variant}>Default</Button>
          <Button color="primary" variant={variant}>
            primary
          </Button>
          <Button color="secondary" variant={variant}>
            secondary
          </Button>
          <Button disabled variant={variant}>
            disabled
          </Button>
          <Button color="primary" disabled variant={variant}>
            disabled primary
          </Button>
          <Button color="secondary" disabled variant={variant}>
            disabled secondary
          </Button>
        </div>
      ))}
    </>
  );
}) as React.FC;
