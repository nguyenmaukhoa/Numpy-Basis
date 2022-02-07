
# <font color='blue'>Numpy Refresher</font>

### <font style="color:rgb(8,133,37)">Why do we need a special library for math and DL?</font>
Python provides data types such as lists / tuples out of the box. Then, why are we using special libraries for deep learning tasks, such as Pytorch or TensorFlow, and not using standard types?

The major reason is efficiency - In pure python, there are no primitive types for numbers, as in e.g. C language. All the data types in Python are objects with lots of properties and methods. You can see it using the `dir` function:


```python
a = 3
dir(a)[-10:]
```




    ['__trunc__',
     '__xor__',
     'bit_length',
     'conjugate',
     'denominator',
     'from_bytes',
     'imag',
     'numerator',
     'real',
     'to_bytes']



### <font style="color:rgb(8,133,37)">Python Issues</font>

- slow in tasks that require tons of simple math operations on numbers
- huge memory overhead due to storing plain numbers as objects
- runtime overhead during memory dereferencing - cache issues


NumPy is an abbreviation for "numerical python" and as it stands from the naming it provides a rich collection of operations on the numerical data types with a python interface. The core data structure of NumPy is `ndarray` - a multidimensional array. Let's take a look at its interface in comparison with plain python lists.

# <font color='blue'>Performance comparison of Numpy array and Python lists </font>

Let's imagine a simple task - we have several 2-dimensional points and we want to represent them as a list of points for further processing. For the sake of simplicity of processing we will not create a `Point` object and will use a list of 2 elements to represent coordinates of each point (`x` and `y`):


```python
# create points list using explicit specification of coordinates of each point
points = [[0, 1], [10, 5], [7, 3]]
points
```




    [[0, 1], [10, 5], [7, 3]]




```python
# create random points
from random import randint

num_dims = 2
num_points = 10
x_range = (0, 10)
y_range = (1, 50)
points = [[randint(*x_range), randint(*y_range)] for _ in range(num_points)]
points
```




    [[7, 27],
     [5, 26],
     [3, 15],
     [1, 19],
     [1, 49],
     [9, 6],
     [1, 1],
     [3, 21],
     [1, 39],
     [8, 42]]



**How can we do the same using Numpy? Easy!**


```python
import numpy as np
points = np.array(points)  # we are able to create numpy arrays from python lists
points
```




    array([[ 7, 27],
           [ 5, 26],
           [ 3, 15],
           [ 1, 19],
           [ 1, 49],
           [ 9,  6],
           [ 1,  1],
           [ 3, 21],
           [ 1, 39],
           [ 8, 42]])




```python
# create random points using numpy library
num_dims = 2
num_points = 10
x_range = (0, 11)
y_range = (1, 51)
points = np.random.randint(
    low=(x_range[0], y_range[0]), high=(x_range[1], y_range[1]), size=(num_points, num_dims)
)
points
```




    array([[ 5, 34],
           [ 4, 27],
           [ 1, 35],
           [ 2, 32],
           [ 1, 10],
           [ 7, 47],
           [ 7, 32],
           [ 0, 20],
           [ 8, 26],
           [ 9,  7]])



**It may look as over-complication to use NumPy for the creation of such a list and we still cannot see the good sides of this approach. But let's take a look at the performance side.**


```python
num_dims = 2
num_points = 100000
x_range = (0, 10)
y_range = (1, 50)
```

### <font style="color:rgb(8,133,37)">Python performance</font>


```python
%timeit \
points = [[randint(*x_range), randint(*y_range)] for _ in range(num_points)]
```

    207 ms ± 337 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


### <font style="color:rgb(8,133,37)">NumPy performance</font>


```python
%timeit \
points = np.random.randint(low=(x_range[0], y_range[0]), high=(x_range[1], y_range[1]), size=(num_points, num_dims))
```

    5.61 ms ± 42.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Wow, NumPy is **around 50 times faster** than pure Python on this task! One may say that the size of the array we're generating is relatively large, but it's very reasonable if we take into account the dimensions of inputs (and weights) in neural networks (or math problems such as hydrodynamics).

# <font style="color:blue">Basics of Numpy </font>
We will go over some of the useful operations of Numpy arrays, which are most commonly used in ML tasks.

## <font color='blue'>1. Basic Operations </font>


### <font style="color:rgb(8,133,37)">1.1. Python list to numpy array</font>


```python
py_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

np_array = np.array(py_list)
np_array
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])




```python
py_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

np_array= np.array(py_list)
np_array
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]])



### <font style="color:rgb(8,133,37)">1.2. Slicing and Indexing</font>


```python
print('First row:\t\t\t{}'.format(np_array[0]))
print('First column:\t\t\t{}'.format(np_array[:, 0]))
print('3rd row 2nd column element:\t{}'.format(np_array[2][1]))
print('2nd onwards row and 2nd onwards column:\n{}'.format(np_array[1:, 1:]))
print('Last 2 rows and last 2 columns:\n{}'.format(np_array[-2:, -2:]))
print('Array with 3rd, 1st and 4th row:\n{}'.format(np_array[[2, 0, 3]]))
```

    First row:			[1 2 3]
    First column:			[ 1  4  7 10]
    3rd row 2nd column element:	8
    2nd onwards row and 2nd onwards column:
    [[ 5  6]
     [ 8  9]
     [11 12]]
    Last 2 rows and last 2 columns:
    [[ 8  9]
     [11 12]]
    Array with 3rd, 1st and 4th row:
    [[ 7  8  9]
     [ 1  2  3]
     [10 11 12]]


### <font style="color:rgb(8,133,37)">1.3. Basic attributes of NumPy array</font>

Get a full list of attributes of an ndarray object [here](https://numpy.org/devdocs/user/quickstart.html).


```python
print('Data type:\t{}'.format(np_array.dtype))
print('Array shape:\t{}'.format(np_array.shape))
```

    Data type:	int64
    Array shape:	(4, 3)


Let's create a function (with name `array_info`) to print the NumPy array, its shape, and its data type. We use this function to print arrays further in this section. 



```python
def array_info(array):
    print('Array:\n{}'.format(array))
    print('Data type:\t{}'.format(array.dtype))
    print('Array shape:\t{}\n'.format(array.shape))
    
array_info(np_array)
```

    Array:
    [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]]
    Data type:	int64
    Array shape:	(4, 3)
    


### <font style="color:rgb(8,133,37)">1.4. Creating NumPy array using built-in functions and datatypes</font>

The full list of supported data types can be found [here](https://numpy.org/devdocs/user/basics.types.html).


**Sequence Array**

`np.arange([start, ]stop, [step, ]dtype=None)`

Return evenly spaced values in `[start, stop)`.

More delatis of the function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html).


```python
# sequence array
array = np.arange(10, dtype=np.int64)
array_info(array)
```

    Array:
    [0 1 2 3 4 5 6 7 8 9]
    Data type:	int64
    Array shape:	(10,)
    



```python
# sequence array
array = np.arange(5, 10, dtype=np.float)
array_info(array)
```

    Array:
    [5. 6. 7. 8. 9.]
    Data type:	float64
    Array shape:	(5,)
    


    /usr/lib/python3.7/site-packages/ipykernel_launcher.py:2: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      


**Zeroes Array**


```python
# Zero array/matrix
zeros = np.zeros((2, 3), dtype=np.float32)
array_info(zeros)
```

    Array:
    [[0. 0. 0.]
     [0. 0. 0.]]
    Data type:	float32
    Array shape:	(2, 3)
    


**Ones Array**


```python
# ones array/matrix
ones = np.ones((3, 2), dtype=np.int8)
array_info(ones)
```

    Array:
    [[1 1]
     [1 1]
     [1 1]]
    Data type:	int8
    Array shape:	(3, 2)
    


**Constant Array**


```python
# constant array/matrix
array = np.full((3, 3), 3.14)
array_info(array)
```

    Array:
    [[3.14 3.14 3.14]
     [3.14 3.14 3.14]
     [3.14 3.14 3.14]]
    Data type:	float64
    Array shape:	(3, 3)
    


**Identity Array**


```python
# identity array/matrix
identity = np.eye(5, dtype=np.float32)      # identity matrix of shape 5x5
array_info(identity)
```

    Array:
    [[1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     [0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1.]]
    Data type:	float32
    Array shape:	(5, 5)
    


**Random Integers Array**

`np.random.randint(low, high=None, size=None, dtype='l')`

Return random integer from the `discrete uniform` distribution in `[low, high)`. If high is `None`, then return elements are in `[0, low)`

More details can be found [here](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randint.html).


```python
# random integers array/matrix
rand_int = np.random.randint(5, 10, (2,3)) # random integer array of shape 2x3, values lies in [5, 10)
array_info(rand_int)
```

    Array:
    [[7 5 6]
     [5 5 6]]
    Data type:	int64
    Array shape:	(2, 3)
    


**Random Array**

`np.random.random(size=None)`

Results are from the `continuous uniform` distribution in `[0.0, 1.0)`.

These types of functions are useful is initializing the weight in Deep Learning. More details and similar functions can found [here](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.random.html).


```python
# random array/matrix
random_array = np.random.random((5, 5))   # random array of shape 5x5
array_info(random_array)
```

    Array:
    [[0.62883151 0.07215295 0.92704924 0.45202991 0.10030216]
     [0.81954221 0.72616345 0.17389839 0.51720665 0.04614688]
     [0.07333179 0.6605047  0.40196895 0.55790114 0.82523577]
     [0.85207596 0.28571834 0.23794541 0.22939261 0.8876192 ]
     [0.25827468 0.50216683 0.82364174 0.2563899  0.13183284]]
    Data type:	float64
    Array shape:	(5, 5)
    


**Boolean Array**

If we compare above `random_array` with some `constant` or `array` of the same shape, we will get a boolean array.


```python
# Boolean array/matrix
bool_array = random_array > 0.5
array_info(bool_array)
```

    Array:
    [[ True False  True False False]
     [ True  True False  True False]
     [False  True False  True  True]
     [ True False False False  True]
     [False  True  True False False]]
    Data type:	bool
    Array shape:	(5, 5)
    


The boolean array can be used to get value from the array. If we use a boolean array of the same shape as indices, we will get those values for which the boolean array is True, and other values will be masked.

Let's use the above `boolen_array` to get values from `random_array`.


```python
# Use boolean array/matrix to get values from array/matrix
values = random_array[bool_array]
array_info(values)
```

    Array:
    [0.62883151 0.92704924 0.81954221 0.72616345 0.51720665 0.6605047
     0.55790114 0.82523577 0.85207596 0.8876192  0.50216683 0.82364174]
    Data type:	float64
    Array shape:	(12,)
    


Basically, from the above method, we are filtering values that are greater than `0.5`. 

**Linespace**

`np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`

Returns num evenly spaced samples, calculated over the interval `[start, stop]`.

More detais about the function find [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)


```python
# Linspace
linespace = np.linspace(0, 5, 7, dtype=np.float32)   # 7 elements between 0 and 5
array_info(linespace)
```

    Array:
    [0.        0.8333333 1.6666666 2.5       3.3333333 4.1666665 5.       ]
    Data type:	float32
    Array shape:	(7,)
    


### <font style="color:rgb(8,133,37)">1.5. Data type conversion</font>

Sometimes it is essential to convert one data type to another data type.


```python
age_in_years = np.random.randint(0, 100, 10)
array_info(age_in_years)
```

    Array:
    [13 12 84 89 42 70 73 37 98 81]
    Data type:	int64
    Array shape:	(10,)
    


Do we really need an `int64` data type to store age?

So let's convert it to `uint8`.


```python
age_in_years = age_in_years.astype(np.uint8)
array_info(age_in_years)
```

    Array:
    [13 12 84 89 42 70 73 37 98 81]
    Data type:	uint8
    Array shape:	(10,)
    


Let's convert it to `float128`. 😜


```python
age_in_years = age_in_years.astype(np.float128)
array_info(age_in_years)
```

    Array:
    [13. 12. 84. 89. 42. 70. 73. 37. 98. 81.]
    Data type:	float128
    Array shape:	(10,)
    


## <font color='blue'>2. Mathematical functions </font>

Numpy supports a lot of Mathematical operations with array/matrix. Here we will see a few of them which are useful in Deep Learning. All supported functions can be found [here](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html).

### <font style="color:rgb(8,133,37)">2.1. Exponential Function </font>
Exponential functions ( also called `exp` ) are used in neural networks as activations functions. They are used in softmax functions which is widely used in Classification tasks.

Return element-wise `exponential` of `array`.

More details of `np.exp` can be found **[here](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.exp.html#numpy.exp)**


```python
array = np.array([np.full(3, -1), np.zeros(3), np.ones(3)])
array_info(array)

# exponential of a array/matrix
print('Exponential of an array:')
exp_array = np.exp(array)
array_info(exp_array)
```

    Array:
    [[-1. -1. -1.]
     [ 0.  0.  0.]
     [ 1.  1.  1.]]
    Data type:	float64
    Array shape:	(3, 3)
    
    Exponential of an array:
    Array:
    [[0.36787944 0.36787944 0.36787944]
     [1.         1.         1.        ]
     [2.71828183 2.71828183 2.71828183]]
    Data type:	float64
    Array shape:	(3, 3)
    


### <font style="color:rgb(8,133,37)">2.2. Square Root </font>

`np.sqrt` return the element-wise `square-root` (`non-negative`) of an array.

More details of the function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html)

`Root Mean Square Error` (`RMSE`) and `Mean Absolute Error` (`MAE`) commonly used to measure the `accuracy` of continuous variables.


```python
array = np.arange(10)
array_info(array)

print('Square root:')
root_array = np.sqrt(array)
array_info(root_array)
```

    Array:
    [0 1 2 3 4 5 6 7 8 9]
    Data type:	int64
    Array shape:	(10,)
    
    Square root:
    Array:
    [0.         1.         1.41421356 1.73205081 2.         2.23606798
     2.44948974 2.64575131 2.82842712 3.        ]
    Data type:	float64
    Array shape:	(10,)
    


### <font style="color:rgb(8,133,37)">2.3. Logrithm </font>

`np.log` return element-wise natural logrithm of an array.

More details of the function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html)

`Cross-Entropy` / `log loss` is the most commonly used loss in Machine Learning classification problem. 


```python
array = np.array([0, np.e, np.e**2, 1, 10])
array_info(array)

print('Logrithm:')
log_array = np.log(array)
array_info(log_array)
```

    Array:
    [ 0.          2.71828183  7.3890561   1.         10.        ]
    Data type:	float64
    Array shape:	(5,)
    
    Logrithm:
    Array:
    [      -inf 1.         2.         0.         2.30258509]
    Data type:	float64
    Array shape:	(5,)
    


    /usr/lib/python3.7/site-packages/ipykernel_launcher.py:5: RuntimeWarning: divide by zero encountered in log
      """


<font color='red'>**Note:** Getting warning because we are trying to calculate `log(0)`.</font>

### <font style="color:rgb(8,133,37)">2.4. Power </font>

`numpy.power(x1, x2)`

Returns first array elements raised to powers from second array, element-wise.

Second array must be broadcastable to first array.

What is **broadcasting**? We will see later.

More detalis about the function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html)


```python
array = np.arange(0, 6, dtype=np.int64)
array_info(array)

print('Power 3:')
pow_array = np.power(array, 3)
array_info(pow_array)
```

    Array:
    [0 1 2 3 4 5]
    Data type:	int64
    Array shape:	(6,)
    
    Power 3:
    Array:
    [  0   1   8  27  64 125]
    Data type:	int64
    Array shape:	(6,)
    


### <font style="color:rgb(8,133,37)">2.5. Clip Values </font>

`np.clip(a, a_min, a_max)`

Return element-wise cliped values between `a_min` and `a_max`.

More details of the finction can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html)

`Rectified Linear Unit` (`ReLU`) is the most commonly used activation function in Deep Learning.

What ReLU do?

If the value is less than zero, it makes it zero otherwise leave as it is. In NumPy assignment will be implementing this activation function using NumPy.


```python
array = np.random.random((3, 3))
array_info(array)

# clipped between 0.2 and 0.5
print('Clipped between 0.2 and 0.5')
cliped_array = np.clip(array, 0.2, 0.5)
array_info(cliped_array)

# clipped to 0.2
print('Clipped to 0.2')
cliped_array = np.clip(array, 0.2, np.inf)
array_info(cliped_array)
```

    Array:
    [[0.37516209 0.52026875 0.95946665]
     [0.17602981 0.70848208 0.31582605]
     [0.94734138 0.32986771 0.57653725]]
    Data type:	float64
    Array shape:	(3, 3)
    
    Clipped between 0.2 and 0.5
    Array:
    [[0.37516209 0.5        0.5       ]
     [0.2        0.5        0.31582605]
     [0.5        0.32986771 0.5       ]]
    Data type:	float64
    Array shape:	(3, 3)
    
    Clipped to 0.2
    Array:
    [[0.37516209 0.52026875 0.95946665]
     [0.2        0.70848208 0.31582605]
     [0.94734138 0.32986771 0.57653725]]
    Data type:	float64
    Array shape:	(3, 3)
    


## <font color='blue'>3. Reshape ndarray </font>

Reshaping the array / matrix is very often required in Machine Learning and Computer vision. 

### <font style="color:rgb(8,133,37)">3.1. Reshape </font>

`np.reshape` gives an array in new shape, without changing its data.

More details of the function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)


```python
a = np.arange(1, 10, dtype=np.int)
array_info(a)

print('Reshape to 3x3:')
a_3x3 = a.reshape(3, 3)
array_info(a_3x3)

print('Reshape 3x3 to 3x3x1:')
a_3x3x1 = a_3x3.reshape(3, 3, 1)
array_info(a_3x3x1)
```

    Array:
    [1 2 3 4 5 6 7 8 9]
    Data type:	int64
    Array shape:	(9,)
    
    Reshape to 3x3:
    Array:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    Data type:	int64
    Array shape:	(3, 3)
    
    Reshape 3x3 to 3x3x1:
    Array:
    [[[1]
      [2]
      [3]]
    
     [[4]
      [5]
      [6]]
    
     [[7]
      [8]
      [9]]]
    Data type:	int64
    Array shape:	(3, 3, 1)
    


    /usr/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      """Entry point for launching an IPython kernel.


### <font style="color:rgb(8,133,37)">3.2. Expand Dim </font>

`np.expand_dims`

In the last reshape, we have added a new axis. We can use `np.expand_dims` or `np.newaxis` to do the same thing.

Mode details for `np.expand_dim` can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html)


```python
print('Using np.expand_dims:')
a_expand = np.expand_dims(a_3x3, axis=2)
array_info(a_expand)

print('Using np.newaxis:')
a_newaxis = a_3x3[..., np.newaxis]
# or 
# a_newaxis = a_3x3[:, :, np.newaxis]
array_info(a_newaxis)
```

    Using np.expand_dims:
    Array:
    [[[1]
      [2]
      [3]]
    
     [[4]
      [5]
      [6]]
    
     [[7]
      [8]
      [9]]]
    Data type:	int64
    Array shape:	(3, 3, 1)
    
    Using np.newaxis:
    Array:
    [[[1]
      [2]
      [3]]
    
     [[4]
      [5]
      [6]]
    
     [[7]
      [8]
      [9]]]
    Data type:	int64
    Array shape:	(3, 3, 1)
    


### <font style="color:rgb(8,133,37)">3.3. Squeeze </font>

Sometimes we need to remove the redundant axis (single-dimensional entries). We can use `np.squeeze` to do this.

More details of `np.squeeze` can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html)

Deep Learning very often uses this functionality.


```python
print('Squeeze along axis=2:')
a_squeezed = np.squeeze(a_newaxis, axis=2)
array_info(a_squeezed)

# should get value error
print('Squeeze along axis=1, should get ValueError')
a_squeezed_error = np.squeeze(a_newaxis, axis=1)  # Getting error because of the size of 
                                                  # axis-1 is not equal to one.
```

    Squeeze along axis=2:
    Array:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    Data type:	int64
    Array shape:	(3, 3)
    
    Squeeze along axis=1, should get ValueError



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-35-e96f455539e0> in <module>
          5 # should get value error
          6 print('Squeeze along axis=1, should get ValueError')
    ----> 7 a_squeezed_error = np.squeeze(a_newaxis, axis=1)  # Getting error because of the size of
          8                                                   # axis-1 is not equal to one.


    <__array_function__ internals> in squeeze(*args, **kwargs)


    /usr/lib/python3.7/site-packages/numpy/core/fromnumeric.py in squeeze(a, axis)
       1506         return squeeze()
       1507     else:
    -> 1508         return squeeze(axis=axis)
       1509 
       1510 


    ValueError: cannot select an axis to squeeze out which has size not equal to one


<font color='red'>**Note:** Getting error because of the size of axis-1 is not equal to one.</font>

### <font style="color:rgb(8,133,37)">3.4. Reshape revisit </font>

We have a 1-d array of length n. We want to reshape in a 2-d array such that the number of columns becomes two, and we do not care about the number of rows. 


```python
a = np.arange(10)
array_info(a)

print('Reshape such that number of column is 2:')
a_col_2 = a.reshape(-1, 2)
array_info(a_col_2)
```

    Array:
    [0 1 2 3 4 5 6 7 8 9]
    Data type:	int64
    Array shape:	(10,)
    
    Reshape such that number of column is 2:
    Array:
    [[0 1]
     [2 3]
     [4 5]
     [6 7]
     [8 9]]
    Data type:	int64
    Array shape:	(5, 2)
    


## <font color='blue'>4. Combine Arrays / Matrix </font>

Combining two or more arrays is a frequent operation in machine learning. Let's have a look at a few methods. 


### <font style="color:rgb(8,133,37)">4.1. Concatenate </font>

`np.concatenate`, Join a sequence of arrays along an existing axis.

More details of the function find [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html)


```python
a1 = np.array([[1, 2, 3], [4, 5, 6]])
a2 = np.array([[7, 8, 9]])

print('Concatenate along axis zero:')
array = np.concatenate((a1, a2), axis=0)
array_info(array)
```

    Concatenate along axis zero:
    Array:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    Data type:	int64
    Array shape:	(3, 3)
    


### <font style="color:rgb(8,133,37)">4.2. hstack </font>

`np.hstack`, stack arrays in sequence horizontally (column-wise).

More details of the function find [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack)


```python
a1 = np.array((1, 2, 3))
a2 = np.array((4, 5, 6))
a_hstacked = np.hstack((a1,a2))

print('Horizontal stack:')
array_info(a_hstacked)
```

    Horizontal stack:
    Array:
    [1 2 3 4 5 6]
    Data type:	int64
    Array shape:	(6,)
    



```python
a1 = np.array([[1],[2],[3]])
a2 = np.array([[4],[5],[6]])
a_hstacked = np.hstack((a1,a2))

print('Horizontal stack:')
array_info(a_hstacked)
```

    Horizontal stack:
    Array:
    [[1 4]
     [2 5]
     [3 6]]
    Data type:	int64
    Array shape:	(3, 2)
    


### <font style="color:rgb(8,133,37)">4.3. vstack </font>

`np.vstack`, tack arrays in sequence vertically (row-wise).

More details of the function find [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html#numpy.vstack)


```python
a1 = np.array([1, 2, 3])
a2 = np.array([4, 5, 6])
a_vstacked = np.vstack((a1, a2))

print('Vertical stack:')
array_info(a_vstacked)
```

    Vertical stack:
    Array:
    [[1 2 3]
     [4 5 6]]
    Data type:	int64
    Array shape:	(2, 3)
    



```python
a1 = np.array([[1, 11], [2, 22], [3, 33]])
a2 = np.array([[4, 44], [5, 55], [6, 66]])
a_vstacked = np.vstack((a1, a2))

print('Vertical stack:')
array_info(a_vstacked)
```

    Vertical stack:
    Array:
    [[ 1 11]
     [ 2 22]
     [ 3 33]
     [ 4 44]
     [ 5 55]
     [ 6 66]]
    Data type:	int64
    Array shape:	(6, 2)
    


## <font color='blue'>5. Element wise Operations </font>


Let's generate a random number to show element-wise operations. 


```python
a = np.random.random((4,4))
b = np.random.random((4,4))
array_info(a)
array_info(b)
```

    Array:
    [[0.894448   0.11701993 0.83289792 0.88351847]
     [0.12202474 0.81966345 0.10307595 0.05559593]
     [0.15736219 0.81569787 0.83315408 0.10474461]
     [0.52216884 0.91130842 0.22294228 0.18338735]]
    Data type:	float64
    Array shape:	(4, 4)
    
    Array:
    [[0.93018548 0.97919687 0.61394268 0.98850134]
     [0.77108645 0.87930347 0.07449472 0.45154363]
     [0.77151181 0.58612289 0.69352912 0.92592587]
     [0.50647278 0.6377183  0.84249894 0.93690901]]
    Data type:	float64
    Array shape:	(4, 4)
    


### <font style="color:rgb(8,133,37)">5.1. Element wise Scalar Operation </font>

**Scalar Addition**


```python
a + 5 # element wise scalar addition
```




    array([[5.894448  , 5.11701993, 5.83289792, 5.88351847],
           [5.12202474, 5.81966345, 5.10307595, 5.05559593],
           [5.15736219, 5.81569787, 5.83315408, 5.10474461],
           [5.52216884, 5.91130842, 5.22294228, 5.18338735]])



**Scalar Subtraction**


```python
a - 5 # element wise scalar subtraction
```




    array([[-4.105552  , -4.88298007, -4.16710208, -4.11648153],
           [-4.87797526, -4.18033655, -4.89692405, -4.94440407],
           [-4.84263781, -4.18430213, -4.16684592, -4.89525539],
           [-4.47783116, -4.08869158, -4.77705772, -4.81661265]])



**Scalar Multiplication**


```python
a * 10 # element wise scalar multiplication
```




    array([[8.94447997, 1.17019931, 8.32897921, 8.83518472],
           [1.22024736, 8.19663451, 1.03075947, 0.55595927],
           [1.57362194, 8.1569787 , 8.33154083, 1.04744607],
           [5.22168839, 9.11308418, 2.22942279, 1.83387353]])



**Scalar Division**


```python
a/10 # element wise scalar division
```




    array([[0.0894448 , 0.01170199, 0.08328979, 0.08835185],
           [0.01220247, 0.08196635, 0.01030759, 0.00555959],
           [0.01573622, 0.08156979, 0.08331541, 0.01047446],
           [0.05221688, 0.09113084, 0.02229423, 0.01833874]])



### <font style="color:rgb(8,133,37)">5.2. Element wise Array Operations </font>

**Arrays Addition**


```python
a + b # element wise array/vector addition
```




    array([[1.82463348, 1.09621681, 1.4468406 , 1.87201981],
           [0.89311119, 1.69896692, 0.17757066, 0.50713956],
           [0.92887401, 1.40182076, 1.5266832 , 1.03067047],
           [1.02864162, 1.54902671, 1.06544122, 1.12029637]])



**Arrays Subtraction**


```python
a - b # element wise array/vector subtraction
```




    array([[-0.03573748, -0.86217694,  0.21895525, -0.10498286],
           [-0.64906171, -0.05964002,  0.02858123, -0.3959477 ],
           [-0.61414962,  0.22957498,  0.13962496, -0.82118126],
           [ 0.01569606,  0.27359012, -0.61955666, -0.75352166]])



**Arrays Multiplication**


```python
a * b # element wise array/vector multiplication
```




    array([[0.83200254, 0.11458555, 0.51135158, 0.87335919],
           [0.09409162, 0.72073292, 0.00767861, 0.02510399],
           [0.12140679, 0.47809919, 0.57781662, 0.09698574],
           [0.2644643 , 0.58115805, 0.18782863, 0.17181726]])



**Arrays Division**


```python
a / b # element wise array/vector division
```




    array([[0.96158026, 0.11950603, 1.35663793, 0.89379593],
           [0.1582504 , 0.93217357, 1.3836679 , 0.12312415],
           [0.20396602, 1.39168404, 1.20132531, 0.11312418],
           [1.03099093, 1.42901407, 0.26462025, 0.19573657]])



We can notice that the dimension of both arrays is equal in above arrays element-wise operations. **What if dimensions are not equal.** Let's check!!


```python
print('Array "a":')
array_info(a)
print('Array "c":')
c = np.random.rand(2, 2)
array_info(c)
# Should throw ValueError
a + c
```

    Array "a":
    Array:
    [[0.894448   0.11701993 0.83289792 0.88351847]
     [0.12202474 0.81966345 0.10307595 0.05559593]
     [0.15736219 0.81569787 0.83315408 0.10474461]
     [0.52216884 0.91130842 0.22294228 0.18338735]]
    Data type:	float64
    Array shape:	(4, 4)
    
    Array "c":
    Array:
    [[0.17941534 0.87659001]
     [0.708542   0.11629358]]
    Data type:	float64
    Array shape:	(2, 2)
    



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-51-0de9e8760d46> in <module>
          5 array_info(c)
          6 # Should throw ValueError
    ----> 7 a + c
    

    ValueError: operands could not be broadcast together with shapes (4,4) (2,2) 


<font color='red'>**Oh got the ValueError!!**</font>

What is this error?

<font color='red'>ValueError</font>: operands could not be broadcast together with shapes `(4,4)` `(2,2)` 

**Let's see it next.**


### <font style="color:rgb(8,133,37)">5.3. Broadcasting </font>

There is a concept of broadcasting in NumPy, which tries to copy rows or columns in the lower-dimensional array to make an equal dimensional array of higher-dimensional array. 

Let's try to understand with a simple example.


```python
a = np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9]])
b = np.array([0, 1, 0])

print('Array "a":')
array_info(a)
print('Array "b":')
array_info(b)

print('Array "a+b":')
array_info(a+b)  # b is reshaped such that it can be added to a.


# b = [0,1,0] is broadcasted to     [[0, 1, 0],
#                                    [0, 1, 0],
#                                    [0, 1, 0]]  and added to a.
```

    Array "a":
    Array:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    Data type:	int64
    Array shape:	(3, 3)
    
    Array "b":
    Array:
    [0 1 0]
    Data type:	int64
    Array shape:	(3,)
    
    Array "a+b":
    Array:
    [[1 3 3]
     [4 6 6]
     [7 9 9]]
    Data type:	int64
    Array shape:	(3, 3)
    


## <font color='blue'>6. Linear Algebra</font>

Here we see commonly use linear algebra operations in Machine Learning. 

### <font style="color:rgb(8,133,37)">6.1. Transpose </font>


```python
a = np.random.random((2,3))
print('Array "a":')
array_info(a)

print('Transose of "a":')
a_transpose = a.transpose()
array_info(a_transpose)
```

    Array "a":
    Array:
    [[0.77625501 0.69619367 0.87855923]
     [0.57858181 0.05237765 0.32425545]]
    Data type:	float64
    Array shape:	(2, 3)
    
    Transose of "a":
    Array:
    [[0.77625501 0.57858181]
     [0.69619367 0.05237765]
     [0.87855923 0.32425545]]
    Data type:	float64
    Array shape:	(3, 2)
    


### <font style="color:rgb(8,133,37)">6.2. Matrix Multiplication</font>
We will discuss 2 ways of performing Matrix Multiplication.

- `matmul`
- Python `@` operator

**Using matmul function in numpy**
This is the most used approach for multiplying two matrices using Numpy. [See docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html)


```python
a = np.random.random((3, 4))
b = np.random.random((4, 2))

print('Array "a":')
array_info(a)
print('Array "b"')
array_info(b)

c = np.matmul(a,b) # matrix multiplication of a and b

print('matrix multiplication of a and b:')
array_info(c)

print('{} x {} --> {}'.format(a.shape, b.shape, c.shape)) # dim1 of a and dim0 of b has to be 
                                                        # same for matrix multiplication
```

    Array "a":
    Array:
    [[9.90725517e-01 6.87469209e-01 8.86002819e-01 6.73534080e-01]
     [2.18174771e-02 4.64527225e-01 9.62589048e-01 8.68540236e-01]
     [9.74114355e-01 5.75293039e-04 3.43527156e-01 4.68362839e-01]]
    Data type:	float64
    Array shape:	(3, 4)
    
    Array "b"
    Array:
    [[0.36668823 0.65516197]
     [0.60588256 0.28107264]
     [0.16075518 0.04400326]
     [0.36617621 0.66253226]]
    Data type:	float64
    Array shape:	(4, 2)
    
    matrix multiplication of a and b:
    Array:
    [[1.16887468 1.32753953]
     [0.7622291  0.76265285]
     [0.58427192 0.96378618]]
    Data type:	float64
    Array shape:	(3, 2)
    
    (3, 4) x (4, 2) --> (3, 2)


**Using `@` operator**
This method of multiplication was introduced in Python 3.5. [See docs](https://www.python.org/dev/peps/pep-0465/)


```python
a = np.random.random((3, 4))
b = np.random.random((4, 2))

print('Array "a":')
array_info(a)
print('Array "b"')
array_info(b)

c = a@b # matrix multiplication of a and b
array_info(c)
```

    Array "a":
    Array:
    [[0.30409247 0.08228474 0.31074426 0.24956204]
     [0.03615212 0.34834705 0.39850674 0.96835052]
     [0.27071512 0.91869934 0.02577805 0.3409815 ]]
    Data type:	float64
    Array shape:	(3, 4)
    
    Array "b"
    Array:
    [[0.75468227 0.63456745]
     [0.32153786 0.69233732]
     [0.54142343 0.22115378]
     [0.98725984 0.4156632 ]]
    Data type:	float64
    Array shape:	(4, 2)
    
    Array:
    [[0.67057766 0.422392  ]
     [1.31106459 0.75475357]
     [0.85029471 0.95527122]]
    Data type:	float64
    Array shape:	(3, 2)
    


### <font style="color:rgb(8,133,37)">6.3. Inverse</font>


```python
A = np.random.random((3,3))
print('Array "A":')
array_info(A)
A_inverse = np.linalg.inv(A)
print('Inverse of "A" ("A_inverse"):')
array_info(A_inverse)

print('"A x A_inverse = Identity" should be true:')
A_X_A_inverse = np.matmul(A, A_inverse)  # A x A_inverse = I = Identity matrix
array_info(A_X_A_inverse)
```

    Array "A":
    Array:
    [[0.99520057 0.93468447 0.23068449]
     [0.52396588 0.9678274  0.94871991]
     [0.92673353 0.50551948 0.57035442]]
    Data type:	float64
    Array shape:	(3, 3)
    
    Inverse of "A" ("A_inverse"):
    Array:
    [[ 0.15448223 -0.88856837  1.41555114]
     [ 1.23820119  0.75490115 -1.75649302]
     [-1.34845748  0.77469175  1.0100785 ]]
    Data type:	float64
    Array shape:	(3, 3)
    
    "A x A_inverse = Identity" should be true:
    Array:
    [[ 1.00000000e+00  6.19530067e-17  1.10554200e-16]
     [ 9.82717171e-17  1.00000000e+00 -8.58257970e-17]
     [-8.35408602e-17  3.25387341e-18  1.00000000e+00]]
    Data type:	float64
    Array shape:	(3, 3)
    


### <font style="color:rgb(8,133,37)">6.4. Dot Product</font>


```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

dot_pro = np.dot(a, b)  # It will be a scalar, so its shape will be empty
array_info(dot_pro)
```

    Array:
    70
    Data type:	int64
    Array shape:	()
    


## <font color='blue'>7. Array statistics</font>

### <font style="color:rgb(8,133,37)">7.1. Sum</font>


```python
a = np.array([1, 2, 3, 4, 5])

print(a.sum())
```

    15


### <font style="color:rgb(8,133,37)">7.2. Sum along Axis</font>


```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print('')

print('sum along 0th axis = ',a.sum(axis = 0)) # sum along 0th axis ie: 1+4, 2+5, 3+6
print("")
print('sum along 1st axis = ',a.sum(axis = 1)) # sum along 1st axis ie: 1+2+3, 4+5+6
```

    [[1 2 3]
     [4 5 6]]
    
    sum along 0th axis =  [5 7 9]
    
    sum along 1st axis =  [ 6 15]


### <font style="color:rgb(8,133,37)">7.3. Minimum and Maximum</font>


```python
a = np.array([-1.1, 2, 5, 100])

print('Minimum = ', a.min())
print('Maximum = ', a.max())
```

    Minimum =  -1.1
    Maximum =  100.0


### <font style="color:rgb(8,133,37)">7.4. Min and Max along Axis</font>


```python
a = np.array([[-2, 0, 2], [1, 2, 3]])

print('a =\n',a,'\n')
print('Minimum = ', a.min())
print('Maximum = ', a.max())
print()
print('Minimum along axis 0 = ', a.min(0))
print('Maximum along axis 0 = ', a.max(0))
print()
print('Minimum along axis 1 = ', a.min(1))
print('Maximum along axis 1 = ', a.max(1))
```

    a =
     [[-2  0  2]
     [ 1  2  3]] 
    
    Minimum =  -2
    Maximum =  3
    
    Minimum along axis 0 =  [-2  0  2]
    Maximum along axis 0 =  [1 2 3]
    
    Minimum along axis 1 =  [-2  1]
    Maximum along axis 1 =  [2 3]


### <font style="color:rgb(8,133,37)">7.5. Mean and Standard Deviation</font>


```python
a = np.array([-1, 0, -0.4, 1.2, 1.43, -1.9, 0.66])

print('mean of the array = ',a.mean())
print('standard deviation of the array = ',a.std())
```

    mean of the array =  -0.001428571428571414
    standard deviation of the array =  1.1142252730860458


### <font style="color:rgb(8,133,37)">7.6. Standardizing the Array</font>

Make distribution of array elements such that`mean=0` and `std=1`.


```python
a = np.array([-1, 0, -0.4, 1.2, 1.43, -1.9, 0.66])

print('mean of the array = ',a.mean())
print('standard deviation of the array = ',a.std())
print()

standardized_a = (a - a.mean())/a.std()
print('Standardized Array = ', standardized_a)
print()

print('mean of the standardized array = ',standardized_a.mean()) # close to 0
print('standard deviation of the standardized  array = ',standardized_a.std()) # equals to 1
```

    mean of the array =  -0.001428571428571414
    standard deviation of the array =  1.1142252730860458
    
    Standardized Array =  [-8.96202458e-01  1.28212083e-03 -3.57711711e-01  1.07826362e+00
      1.28468507e+00 -1.70393858e+00  5.93621943e-01]
    
    mean of the standardized array =  -3.172065784643304e-17
    standard deviation of the standardized  array =  1.0


# <font color='blue'>References </font>

https://numpy.org/devdocs/user/quickstart.html

https://numpy.org/devdocs/user/basics.types.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html

https://coolsymbol.com/emojis/emoji-for-copy-and-paste.html

https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html

https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.exp.html#numpy.exp

https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack

https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html#numpy.vstack


```python

```
