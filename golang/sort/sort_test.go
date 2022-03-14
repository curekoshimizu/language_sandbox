package sort_test

import (
	"math/rand"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func randomDate() time.Time {
	min := time.Date(2000, 1, 0, 0, 0, 0, 0, time.UTC).Unix()
	max := time.Date(2022, 1, 0, 0, 0, 0, 0, time.UTC).Unix()
	delta := max - min

	sec := rand.Int63n(delta) + min
	return time.Unix(sec, 0)
}

func TestSortTime(t *testing.T) {
	length := 100
	slice := make([]time.Time, 0, length)
	for i := 0; i < length; i++ {
		slice = append(slice, randomDate())
	}

	// sort
	sort.Slice(slice, func(i, j int) bool {
		return slice[i].Before(slice[j])
	})

	// check
	// slice[0] is older than slice[length-1]
	// example.
	//   slice[0]        : 2000-01-08 06:27:00 +0900 JST
	//   slice[length-1] : 2000-02-02 02:18:39 +0900 JST
	assert.True(t, slice[0].Before(slice[length-1]))
}

type Tuple struct {
	X int
	Y float64
}

func randomTuple() Tuple {
	return Tuple{
		X: rand.Int(),
		Y: float64(rand.Int()),
	}
}

func TestStructSort(t *testing.T) {
	length := 100
	slice := make([]Tuple, 0, length)
	for i := 0; i < length; i++ {
		slice = append(slice, randomTuple())
	}

	satisfiedFunc := func(i, j int) bool {
		if slice[i].X == slice[j].X {
			return slice[i].Y < slice[j].Y
		}
		return slice[i].X < slice[j].X
	}
	// sort
	sort.Slice(slice, satisfiedFunc)

	// check
	assert.True(t, satisfiedFunc(0, length-1))
}
