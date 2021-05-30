import { Provider as ReduxProvider } from 'react-redux';

import { Container, createMuiTheme } from '@material-ui/core';
import CssBaseline from '@material-ui/core/CssBaseline';
import {
  ThemeProvider as MaterialThemeProvider,
  StylesProvider,
} from '@material-ui/styles';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';

import AppBar from './AppBar';
import Button from './Button';
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

  return (
    <MaterialThemeProvider theme={theme}>
      <StyledThemeProvider theme={theme}>
        <CssBaseline />
        <Container>
          <AppBar />

          <Typography />
          <Button />
        </Container>
      </StyledThemeProvider>
    </MaterialThemeProvider>
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
