import { createMuiTheme } from '@material-ui/core';
import {
  ThemeProvider as MaterialThemeProvider,
  StylesProvider,
} from '@material-ui/styles';
import styled, {
  ThemeProvider as StyledThemeProvider,
} from 'styled-components';

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
        <div>Hello</div>;
      </StyledThemeProvider>
    </MaterialThemeProvider>
  </StylesProvider>
);

export default App;
