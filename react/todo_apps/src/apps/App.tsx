import { useState } from 'react';

import {
  Container,
  createMuiTheme,
  Button as MaterialUiButton,
} from '@material-ui/core';
import CssBaseline from '@material-ui/core/CssBaseline';
import {
  ThemeProvider as MaterialThemeProvider,
  StylesProvider,
} from '@material-ui/styles';
import styled, {
  ThemeProvider as StyledThemeProvider,
} from 'styled-components';

import AppBar from './AppBar';
import Button from './Button';
import Typography from './Typography';

const StyledButton = styled(MaterialUiButton)`
  font-size: 2em;
  margin: 1em;
  padding: 0.25em 1em;
  border-radius: 3px;
`;

const App: React.FC = () => {
  const [darkMode, setDarkMode] = useState<boolean>(false);
  const theme = createMuiTheme({
    palette: {
      primary: {
        main: '#fb8a8a',
      },
      type: darkMode ? 'dark' : 'light',
    },
  });

  return (
    <StylesProvider injectFirst>
      <MaterialThemeProvider theme={theme}>
        <StyledThemeProvider theme={theme}>
          <CssBaseline />
          <Container>
            <AppBar />

            <Typography />
            <Button />
            <StyledButton color="primary" variant="contained">
              Contained
            </StyledButton>
            <StyledButton
              color="primary"
              onClick={() => setDarkMode(!darkMode)}
            >
              mode change
            </StyledButton>
          </Container>
        </StyledThemeProvider>
      </MaterialThemeProvider>
    </StylesProvider>
  );
};

export default App;
