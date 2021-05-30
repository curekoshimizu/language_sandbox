import Box from '@material-ui/core/Box';
import {
  amber,
  blue,
  blueGrey,
  brown,
  cyan,
  deepOrange,
  deepPurple,
  green,
  grey,
  indigo,
  lightBlue,
  lightGreen,
  lime,
  orange,
  pink,
  purple,
  red,
  teal,
  yellow,
} from '@material-ui/core/colors';
import Grid from '@material-ui/core/Grid';

const hues = [
  { component: red, name: 'red' },
  { component: pink, name: 'pink' },
  { component: purple, name: 'purple' },
  { component: deepPurple, name: 'deepPurple' },
  { component: indigo, name: 'indigo' },
  { component: blue, name: 'blue' },
  { component: lightBlue, name: 'lightBlue' },
  { component: cyan, name: 'cyan' },
  { component: teal, name: 'teal' },
  { component: green, name: 'green' },
  { component: lightGreen, name: 'lightGreen' },
  { component: lime, name: 'lime' },
  { component: yellow, name: 'yellow' },
  { component: amber, name: 'amber' },
  { component: orange, name: 'orange' },
  { component: deepOrange, name: 'deepOrange' },
  { component: brown, name: 'brown' },
  { component: grey, name: 'grey' },
  { component: blueGrey, name: 'blueGrey' },
];

interface ColorBoxProps {
  backgroundColor: string;
}

const ColorBox: React.FC<ColorBoxProps> = ({ backgroundColor }) => {
  const size = 40;

  return <Box height={size} style={{ backgroundColor }} width={size} />;
};

export default (() => {
  const shades: Array<keyof typeof red> = [
    50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
  ];

  return (
    <Grid container>
      {hues.map((hue) => (
        <Grid item key={hue.name}>
          {shades.map((shade) => (
            <ColorBox
              backgroundColor={hue.component[shade]}
              key={`${hue.name}-${shade}`}
            />
          ))}
        </Grid>
      ))}
    </Grid>
  );
}) as React.FC;
