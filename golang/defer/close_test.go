package defer_test

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func NewCloseObject(openErr error, runErr error, closeErr error) CloseObject {
	return CloseObject{
		openErr:  openErr,
		runErr:   runErr,
		closeErr: closeErr,
	}
}

type CloseObject struct {
	openErr  error
	runErr   error
	closeErr error
}

func (a CloseObject) Open() error {
	return a.openErr
}

func (a CloseObject) Run() error {
	return a.runErr
}

func (a CloseObject) Close() error {
	return a.closeErr
}

func TestClose1(t *testing.T) {
	f := func(obj CloseObject) (retErr error) {
		if retErr = obj.Open(); retErr != nil {
			return
		}
		defer func() {
			if err := obj.Close(); err != nil && retErr == nil {
				retErr = err
			}
		}()
		retErr = obj.Run()
		return
	}

	err1 := errors.New("error1")
	err2 := errors.New("error2")
	assert.Nil(t, f(NewCloseObject(nil, nil, nil)))
	assert.Equal(t, err1, f(NewCloseObject(err1, nil, nil)))
	assert.Equal(t, err1, f(NewCloseObject(nil, err1, nil)))
	assert.Equal(t, err1, f(NewCloseObject(nil, nil, err1)))
	assert.Equal(t, err1, f(NewCloseObject(nil, err1, err2)))
}

func TestClose2(t *testing.T) {
	f := func(obj CloseObject) (retErr error) {
		// ここの Open のところは retErr を使わないという例
		if err := obj.Open(); err != nil {
			return err
		}

		defer func() {
			if err := obj.Close(); err != nil && retErr == nil {
				retErr = err
			}
		}()
		retErr = obj.Run()

		// ここは return でもよいし、 TestClose3 のように return retErr にしてもよい
		return
	}

	err1 := errors.New("error1")
	err2 := errors.New("error2")
	assert.Nil(t, f(NewCloseObject(nil, nil, nil)))
	assert.Equal(t, err1, f(NewCloseObject(err1, nil, nil)))
	assert.Equal(t, err1, f(NewCloseObject(nil, err1, nil)))
	assert.Equal(t, err1, f(NewCloseObject(nil, nil, err1)))
	assert.Equal(t, err1, f(NewCloseObject(nil, err1, err2)))
}

func TestClose3(t *testing.T) {
	f := func(obj CloseObject) (retErr error) {
		if err := obj.Open(); err != nil {
			return err
		}
		defer func() {
			if err := obj.Close(); err != nil && retErr == nil {
				retErr = err
			}
		}()
		retErr = obj.Run()
		return retErr // ここが差分
	}

	err1 := errors.New("error1")
	err2 := errors.New("error2")
	assert.Nil(t, f(NewCloseObject(nil, nil, nil)))
	assert.Equal(t, err1, f(NewCloseObject(err1, nil, nil)))
	assert.Equal(t, err1, f(NewCloseObject(nil, err1, nil)))
	assert.Equal(t, err1, f(NewCloseObject(nil, nil, err1)))
	assert.Equal(t, err1, f(NewCloseObject(nil, err1, err2)))
}

func NewCloseIntObject(openErr error, runErr error, closeErr error) CloseIntObject {
	return CloseIntObject{
		openErr:  openErr,
		runErr:   runErr,
		closeErr: closeErr,
	}
}

type CloseIntObject struct {
	openErr  error
	runErr   error
	closeErr error
}

func (a CloseIntObject) Open() (int, error) {
	return 1, a.openErr
}

func (a CloseIntObject) Run() (int, error) {
	return 2, a.runErr
}

func (a CloseIntObject) Close() error {
	return a.closeErr
}

func TestIntClose1(t *testing.T) {
	f := func(obj CloseIntObject) (retInt int, retErr error) {
		if retInt, retErr = obj.Open(); retErr != nil {
			return
		}
		defer func() {
			if err := obj.Close(); err != nil && retErr == nil {
				retErr = err
			}
		}()
		retInt, retErr = obj.Run()
		return
	}

	err1 := errors.New("error1")
	err2 := errors.New("error2")
	retInt, retErr := f(NewCloseIntObject(nil, nil, nil))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, nil, retErr)
	retInt, retErr = f(NewCloseIntObject(err1, nil, nil))
	assert.Equal(t, 1, retInt)
	assert.Equal(t, err1, retErr)
	retInt, retErr = f(NewCloseIntObject(nil, err1, nil))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, err1, retErr)
	retInt, retErr = f(NewCloseIntObject(nil, nil, err1))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, err1, retErr)
	retInt, retErr = f(NewCloseIntObject(nil, err1, err2))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, err1, retErr)
}

func TestIntClose2(t *testing.T) {
	f := func(obj CloseIntObject) (retInt int, retErr error) {
		// ここの Open のところは retIntとretErr を使わないという例
		if ret, err := obj.Open(); err != nil {
			return ret, err
		}

		defer func() {
			if err := obj.Close(); err != nil && retErr == nil {
				retErr = err
			}
		}()
		retInt, retErr = obj.Run()
		return
	}

	err1 := errors.New("error1")
	err2 := errors.New("error2")
	retInt, retErr := f(NewCloseIntObject(nil, nil, nil))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, nil, retErr)
	retInt, retErr = f(NewCloseIntObject(err1, nil, nil))
	assert.Equal(t, 1, retInt)
	assert.Equal(t, err1, retErr)
	retInt, retErr = f(NewCloseIntObject(nil, err1, nil))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, err1, retErr)
	retInt, retErr = f(NewCloseIntObject(nil, nil, err1))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, err1, retErr)
	retInt, retErr = f(NewCloseIntObject(nil, err1, err2))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, err1, retErr)
}

func TestIntClose3(t *testing.T) {
	f := func(obj CloseIntObject) (retInt int, retErr error) {
		if ret, err := obj.Open(); err != nil {
			return ret, err
		}
		defer func() {
			if err := obj.Close(); err != nil && retErr == nil {
				retErr = err
			}
		}()

		// := でも retErr が新しい変数というわけではないので
		// きちんとreturnできる
		ret2, retErr := obj.Run()
		return ret2, retErr
	}

	err1 := errors.New("error1")
	err2 := errors.New("error2")
	retInt, retErr := f(NewCloseIntObject(nil, nil, nil))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, nil, retErr)
	retInt, retErr = f(NewCloseIntObject(err1, nil, nil))
	assert.Equal(t, 1, retInt)
	assert.Equal(t, err1, retErr)
	retInt, retErr = f(NewCloseIntObject(nil, err1, nil))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, err1, retErr)
	retInt, retErr = f(NewCloseIntObject(nil, nil, err1))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, err1, retErr)
	retInt, retErr = f(NewCloseIntObject(nil, err1, err2))
	assert.Equal(t, 2, retInt)
	assert.Equal(t, err1, retErr)
}
