import { useHistory } from 'react-router-dom';

import AppBar from '@material-ui/core/AppBar';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import { createStyles, makeStyles, Theme } from '@material-ui/core/styles';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Brightness4Icon from '@material-ui/icons/Brightness4';
import MenuIcon from '@material-ui/icons/Menu';

import { actions } from '../reducers/themeSlice';
import { useAppDispatch } from './store';

export interface Links {
  component: React.FC;
  path: string;
  title: string;
}

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      flexGrow: 1,
    },
    menuButton: {
      marginRight: theme.spacing(2),
    },
    title: {
      flexGrow: 1,
    },
  })
);

interface AppBarProps {
  links: Array<Links>;
}

export default (({ links }) => {
  const classes = useStyles();
  const dispatch = useAppDispatch();
  const history = useHistory();

  return (
    <div className={classes.root}>
      <AppBar position="static">
        <Toolbar>
          <IconButton>
            <MenuIcon />
          </IconButton>
          <Typography className={classes.title} variant="h6">
            {links.map((link) => (
              <Button key={link.path} onClick={() => history.push(link.path)}>
                {link.title}
              </Button>
            ))}
          </Typography>
          <IconButton onClick={() => dispatch(actions.changeTheme())}>
            <Brightness4Icon />
          </IconButton>
        </Toolbar>
      </AppBar>
    </div>
  );
}) as React.FC<AppBarProps>;
