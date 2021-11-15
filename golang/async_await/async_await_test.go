package async_await

import (
    "testing"

    "github.com/stretchr/testify/assert"
)


func TestAdd(t *testing.T) {
    assert.Equal(t, Add(1, 2), 3);
}

