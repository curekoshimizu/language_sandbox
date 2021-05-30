import { createMuiTheme } from '@material-ui/core';
import Button from '@material-ui/core/Button';
import CssBaseline from '@material-ui/core/CssBaseline';
import {
  ThemeProvider as MaterialThemeProvider,
  StylesProvider,
} from '@material-ui/styles';
import styled, {
  ThemeProvider as StyledThemeProvider,
} from 'styled-components';

const StyledButton = styled(Button)`
  font-size: 2em;
  margin: 1em;
  padding: 0.25em 1em;
  border-radius: 3px;
`;

const theme = createMuiTheme({
  palette: {
    primary: {
      main: '#8BC34A',
      dark: '#689F38',
      light: '#DCEDC8',
    },
    secondary: {
      main: '#FF5722',
    },
    text: {
      primary: '#212121',
      secondary: '#757575',
    },
  },
});

const App: React.FC = () => (
  <StylesProvider injectFirst>
    <MaterialThemeProvider theme={theme}>
      <StyledThemeProvider theme={theme}>
        <CssBaseline />
        <StyledButton color="primary" variant="contained">
          Default
        </StyledButton>
        <div>Hello</div>
      </StyledThemeProvider>
    </MaterialThemeProvider>
  </StylesProvider>
);

export default App;
