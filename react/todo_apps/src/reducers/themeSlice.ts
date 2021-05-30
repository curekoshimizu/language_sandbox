import { createSlice } from '@reduxjs/toolkit';

export type ThemeState = 'dark' | 'light';

const initialState = 'light' as ThemeState;

const slice = createSlice({
  name: 'theme',
  initialState,
  reducers: {
    changeTheme: (state) => {
      return state === 'dark' ? 'light' : 'dark';
    },
  },
});

export const { actions } = slice;

export default slice.reducer;
