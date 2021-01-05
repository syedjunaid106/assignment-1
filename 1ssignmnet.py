{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arra=np.zeros(10)\n",
    "arra"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,\n",
       "       27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,\n",
       "       44, 45, 46, 47, 48, 49])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrb=np.arange(10,50)\n",
    "arrb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(40,)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrb.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int32')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrb.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.18.1\n",
      "blas_mkl_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/92307/anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/92307/anaconda3\\\\Library\\\\include']\n",
      "blas_opt_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/92307/anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/92307/anaconda3\\\\Library\\\\include']\n",
      "lapack_mkl_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/92307/anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/92307/anaconda3\\\\Library\\\\include']\n",
      "lapack_opt_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/92307/anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/92307/anaconda3\\\\Library\\\\include']\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(np.__version__)\n",
    "print(np.show_config())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrb.ndim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True,  True,  True,  True])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrc=np.ones([5],dtype=bool)\n",
    "arrc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [3, 4, 5]])"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d=np.array([[1,2,3],[3,4,5]])\n",
    "arr2d\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1 2 3]]\n",
      "\n",
      " [[3 4 5]]\n",
      "\n",
      " [[6 7 8]]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d=np.array([[[1,2,3]],[[3,4,5]],[[6,7,8]]])\n",
    "print(arr3d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([9, 8, 7, 6, 5, 4, 3, 2, 1])"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ar=np.arange(1,10)\n",
    "ar=ar[::-1]\n",
    "ar\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrnull=np.zeros(10)\n",
    "arrnull[4]=1\n",
    "arrnull"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 0., 0.],\n",
       "       [0., 1., 0.],\n",
       "       [0., 0., 1.]])"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arriden=np.identity(3)\n",
    "arriden"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 2., 3., 4., 5.])"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrflt= np.array([1, 2, 3, 4, 5],dtype=float)\n",
    "arrflt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.,  8.,  3.],\n",
       "       [28., 10., 72.]])"
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1 = np.array([[1., 2., 3.],\n",
    "\n",
    "            [4., 5., 6.]])  \n",
    "\n",
    "arr2 = np.array([[0., 4., 1.],\n",
    "\n",
    "           [7., 2., 12.]])\n",
    "\n",
    "arr3=arr1*arr2\n",
    "arr3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ True, False,  True],\n",
       "       [False,  True, False]])"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1 = np.array([[1., 2., 3.],\n",
    "\n",
    "            [4., 5., 6.]]) \n",
    "\n",
    "arr2 = np.array([[0., 4., 1.],\n",
    "\n",
    "            [7., 2., 12.]])\n",
    "arr3=np.greater(arr1,arr2)\n",
    "arr3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 3, 5, 7])"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrodd=np.arange(0,9)\n",
    "arrodd=arrodd[arrodd%2==1]\n",
    "arrodd\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'int' object does not support item assignment",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-134-e0c098969bb7>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0marrodd\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0marrodd\u001b[0m\u001b[1;33m%\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m==\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0marrodd\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: 'int' object does not support item assignment"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2,  3,  4, 12, 12, 12, 12,  9])"
      ]
     },
     "execution_count": 109,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrrep = np.arange(10)\n",
    "arrrep[5:9]=12\n",
    "arrrep"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 1., 1.],\n",
       "       [1., 0., 1.],\n",
       "       [1., 1., 1.]])"
      ]
     },
     "execution_count": 143,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d=np.ones((3,3))\n",
    "arr2d[1:-1,1:-1]=0\n",
    "arr2d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3],\n",
       "       [ 4, 12,  6],\n",
       "       [ 7,  8,  9]])"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d = np.array([[1, 2, 3],\n",
    "                  [4, 5, 6], \n",
    "                  [7, 8, 9]])\n",
    "arr2d = np.where(arr2d == 5,12,arr2d)\n",
    "arr2d\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[64, 64, 64],\n",
       "        [ 4,  5,  6]],\n",
       "\n",
       "       [[ 7,  8,  9],\n",
       "        [10, 11, 12]]])"
      ]
     },
     "execution_count": 146,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d = np.array([\n",
    "    [\n",
    "        [1, 2, 3], \n",
    "        [4, 5, 6]\n",
    "    ], \n",
    "    [\n",
    "        [7, 8, 9], \n",
    "        [10, 11, 12]\n",
    "    ]\n",
    "])\n",
    "arr3d[0][0] = 64\n",
    "arr3d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 153,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d=np.array([[0,1,2,3,4],[5,6,7,8,9]])\n",
    "arr3d=arr3d[1]\n",
    "arr3d\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrq=np.array([[0,1,2,3,4],[5,6,7,8,9]])\n",
    "arrq=arrq[1][1]\n",
    "arrq"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 7])"
      ]
     },
     "execution_count": 160,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arrq=np.array([[0,1,2,3,4],[5,6,7,8,9]])\n",
    "arrq=arrq[:,2]\n",
    "arrq"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Minimun Value -1.8918648082216167\n",
      "Maximum Value 2.6074521341662926\n"
     ]
    }
   ],
   "source": [
    "rand = np.random.randn((100)).reshape((10,10))\n",
    "print(f\"Minimun Value {np.amin(rand)}\")\n",
    "print(f\"Maximum Value {np.amax(rand)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Common Elements are : [2 4]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "a = np.array([1,2,3,2,3,4,3,4,5,6]) \n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "print(f\"Common Elements are : {np.intersect1d(a,b)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 5], dtype=int64)"
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.searchsorted(a, np.intersect1d(a, b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-1.31206273 -1.06803045 -0.40240925  0.99722411]\n",
      " [-1.68865383  0.18978844 -0.04037382  1.74365814]\n",
      " [ 0.32985448  0.17177463  0.80181941 -0.1415496 ]\n",
      " [-0.61168986 -0.32555532 -0.80235934 -1.03452641]\n",
      " [ 1.29412392 -0.736921    0.68626656 -0.71691332]\n",
      " [-1.10559701  1.3037042   0.85944044  1.47989225]\n",
      " [-0.98362156  1.59204842 -0.20993803  0.42310396]]\n",
      "==================================================\n",
      "[[-1.31206273 -1.06803045 -0.40240925  0.99722411]\n",
      " [-1.68865383  0.18978844 -0.04037382  1.74365814]\n",
      " [-0.61168986 -0.32555532 -0.80235934 -1.03452641]\n",
      " [-1.10559701  1.3037042   0.85944044  1.47989225]\n",
      " [-0.98362156  1.59204842 -0.20993803  0.42310396]]\n"
     ]
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) \n",
    "data = np.random.randn(7, 4)\n",
    "print(data)\n",
    "print('==================================================')\n",
    "print(data[names != 'Will'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-1.31206273 -1.06803045 -0.40240925  0.99722411]\n",
      " [-1.68865383  0.18978844 -0.04037382  1.74365814]\n",
      " [ 0.32985448  0.17177463  0.80181941 -0.1415496 ]\n",
      " [-0.61168986 -0.32555532 -0.80235934 -1.03452641]\n",
      " [ 1.29412392 -0.736921    0.68626656 -0.71691332]\n",
      " [-1.10559701  1.3037042   0.85944044  1.47989225]\n",
      " [-0.98362156  1.59204842 -0.20993803  0.42310396]]\n",
      "==================================================\n",
      "[[-1.31206273 -1.06803045 -0.40240925  0.99722411]\n",
      " [-0.61168986 -0.32555532 -0.80235934 -1.03452641]]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print(data)\n",
    "print('==================================================')\n",
    "mask = (names != 'Will') & (names!= 'Joe')\n",
    "print(data[mask])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 4.0763989   3.39975572  6.05097622]\n",
      " [13.87297355 13.10134293 11.55279846]\n",
      " [ 9.44727646 11.4382458   6.70018448]\n",
      " [ 5.50383394 10.56336176  7.52123257]\n",
      " [ 2.50349737  3.59880723 14.59127857]]\n"
     ]
    }
   ],
   "source": [
    "rand_arr = np.random.uniform(1,15, size=(5,3))\n",
    "print(rand_arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[15.95623613  5.05694447  7.32924862  9.78016017]\n",
      "  [ 9.39788879  4.373588   13.68855901  3.0795141 ]]\n",
      "\n",
      " [[ 2.62441431  9.30086795  1.5547245   8.15955472]\n",
      "  [15.18538673  2.19653503  1.07234622 11.35803492]]]\n"
     ]
    }
   ],
   "source": [
    "rand_arr2 = np.random.uniform(1,16, size=(2,2,4))\n",
    "print(rand_arr2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original Array\n",
      "[[[15.95623613  5.05694447  7.32924862  9.78016017]\n",
      "  [ 9.39788879  4.373588   13.68855901  3.0795141 ]]\n",
      "\n",
      " [[ 2.62441431  9.30086795  1.5547245   8.15955472]\n",
      "  [15.18538673  2.19653503  1.07234622 11.35803492]]]\n",
      "Swapped Axes\n",
      "[[[15.95623613  2.62441431]\n",
      "  [ 9.39788879 15.18538673]]\n",
      "\n",
      " [[ 5.05694447  9.30086795]\n",
      "  [ 4.373588    2.19653503]]\n",
      "\n",
      " [[ 7.32924862  1.5547245 ]\n",
      "  [13.68855901  1.07234622]]\n",
      "\n",
      " [[ 9.78016017  8.15955472]\n",
      "  [ 3.0795141  11.35803492]]]\n"
     ]
    }
   ],
   "source": [
    "print(\"Original Array\")\n",
    "print(rand_arr2)\n",
    "print(\"Swapped Axes\")\n",
    "print(np.swapaxes(rand_arr2, 2, 0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[4 3 3 3 2 4 1 3 3 3]\n"
     ]
    }
   ],
   "source": [
    "arr = np.random.uniform(0,20, size=(10))\n",
    "arr = arr.astype('int32') \n",
    "arr =np.sqrt(arr)\n",
    "arr = np.where(arr < 0.5, 0, arr)\n",
    "arr = arr.astype('int32')\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Array 1 : [ 9.24428492 19.20507242  0.89618599 18.99747282  3.77318696  3.23220402\n",
      "  5.34369148 11.48518066  7.43168054  3.34120514]\n",
      "Array 2 : [15.98613379  4.6740225  16.04451475 10.60173719 10.44513161  6.16575358\n",
      " 16.56438594 15.96133785  6.23318067  6.03292263]\n",
      "New Array : [15.98613379 19.20507242 16.04451475 18.99747282 10.44513161  6.16575358\n",
      " 16.56438594 15.96133785  7.43168054  6.03292263]\n"
     ]
    }
   ],
   "source": [
    "arr = np.random.uniform(0,20, size=(10))\n",
    "arr1 = np.random.uniform(0,20, size=(10))\n",
    "newArr = np.maximum(arr,arr1)\n",
    "print(f\"Array 1 : {arr}\")\n",
    "print(f\"Array 2 : {arr1}\")\n",
    "print(f\"New Array : {newArr}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Bob' 'Joe' 'Will']\n"
     ]
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "print(np.sort(np.unique(names)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3 4]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "a = np.array([1,2,3,4,5]) \n",
    "b = np.array([5,6,7,8,9])\n",
    "c = np.setdiff1d(a, b)\n",
    "print(c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[34 43 10]\n",
      " [82 22 10]\n",
      " [53 94 10]]\n"
     ]
    }
   ],
   "source": [
    "sampleArray = np.array([\n",
    "    [34,43,73],\n",
    "    [82,22,12],\n",
    "    [53,94,66]\n",
    "])\n",
    "newColumn = np.array([[10,10,10]])\n",
    "sampleArray = np.delete(sampleArray, 2, axis=1)\n",
    "sampleArray = np.insert(sampleArray, 2, newColumn, axis=1)\n",
    "print(sampleArray)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 28.  64.]\n",
      " [ 67. 181.]]\n"
     ]
    }
   ],
   "source": [
    "x = np.array([[1., 2., 3.], [4., 5., 6.]]) \n",
    "y = np.array([[6., 23.], [-1, 7], [8, 9]])\n",
    "print(np.dot(x,y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 15  24  38  47  62  71  88  97 108 126 139 144 157 164 181 192 200 201\n",
      " 211 217]\n"
     ]
    }
   ],
   "source": [
    "arr = np.random.uniform(1,20,size=(4,5))\n",
    "arr = arr.astype('int32')\n",
    "print(arr.cumsum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
