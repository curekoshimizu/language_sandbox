import { Provider as ReduxProvider } from 'react-redux';
import { BrowserRouter, Redirect, Route, Switch } from 'react-router-dom';

import { Container, createMuiTheme } from '@material-ui/core';
import CssBaseline from '@material-ui/core/CssBaseline';
import {
  ThemeProvider as MaterialThemeProvider,
  StylesProvider,
} from '@material-ui/styles';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';

import AppBar, { Links } from './AppBar';
import Button from './Button';
import Color from './Color';
import Size from './Size';
import { store, useAppSelector } from './store';
import Typography from './Typography';

const AppMain: React.FC = () => {
  const currentTheme = useAppSelector((state) => state.theme);
  const theme = createMuiTheme({
    palette: {
      primary: {
        main: '#fb8a8a',
      },
      type: currentTheme,
    },
  });

  const links: Array<Links> = [
    { component: Button, path: '/button', title: 'Button' },
    { component: Typography, path: '/typography', title: 'Typography' },
    { component: Color, path: '/color', title: 'Color' },
    { component: Size, path: '/size', title: 'Size' },
  ];

  return (
    <BrowserRouter>
      <MaterialThemeProvider theme={theme}>
        <StyledThemeProvider theme={theme}>
          <CssBaseline />
          <Container>
            <AppBar links={links} />
            <main
              style={{
                padding: theme.spacing(2),
              }}
            >
              <Switch>
                {links.map((link) => {
                  return (
                    <Route
                      component={link.component}
                      key={link.title}
                      path={link.path}
                    />
                  );
                })}
                <Redirect exact from="/" to={links[0].path} />
              </Switch>
            </main>
          </Container>
        </StyledThemeProvider>
      </MaterialThemeProvider>
    </BrowserRouter>
  );
};

export default (() => {
  return (
    <ReduxProvider store={store}>
      <StylesProvider injectFirst>
        <AppMain />
      </StylesProvider>
    </ReduxProvider>
  );
}) as React.FC;
