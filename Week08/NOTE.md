# 8-16 位运算

## 位运算符

1. 左移 << 高位移除，低位补零
2. 右移 >> 低位移除，高位补零
3. 按位或 | 
4. 按位与 &
5. 按位反 ~
6. 按位异或（相同为零不同为一） ^ 

- 异或也可用**不进位加法**理解，相同为零，不同为一
- 异或操作的一些特点
     - x ^ 0 = x
     - x ^ 1s = ~x    // 注意 1s = ~ 0
     - x ^ (~x) = 1s
     - x ^ x = 0
     - c = a ^ b   a = b ^ c    b = a ^ c
     - a ^ b ^ c = a ^ (b ^ c) = (a ^ b) ^ c

7. 指定位置的位运算

- 将x最右边的n位清零 x & (~0 << n)
- 获取x的第n位值（0或者1）： (x>>n) & 1
- 获得x的第n位的幂值： x & (1<<n)
- 仅将第n位置为1： x | (1<<n)
- 仅将第n位置为0： x &(~(1<<n))
- 将x最高位至第n位（含）清零： x & ((1<<n)-1)

### 实战位运算要点

1. 判断奇偶

- x % 2 == 1 ----> (x&1) == 1
- x % 2 == 0 ----> (x&0) == 0

2. x>>1 ----> x//2
3. x = x & (x-1) 清零最低位的1
4. x & -x 得到最低位的1 （负数在计算机中的表示为：原码---->反码---->补码）
5. x & ~x == 0

## 算数移位与逻辑移位



## 位运算的应用



# 8-17  布隆过滤器 LRU Cache

## bloom filter 布隆过滤器

1. 由一个很长的二进制向量和一系列随机映射函数，布隆过滤器可以用于检测一个元素是否在一个集合中。
2. 优点是空间效率和查询时间都远远超一般算法，缺点是有一定的误识别率和删除困难。
3. 可认为是挡在机器面前的快速缓存

```python
from bitarray import bitarray 
import mmh3 
class BloomFilter: 
	def __init__(self, size, hash_num): 
		self.size = size 
		self.hash_num = hash_num 
		self.bit_array = bitarray(size) 
		self.bit_array.setall(0) 
	def add(self, s): 
		for seed in range(self.hash_num): 
			result = mmh3.hash(s, seed) % self.size 
			self.bit_array[result] = 1 
	def lookup(self, s): 
		for seed in range(self.hash_num): 
			result = mmh3.hash(s, seed) % self.size 
			if self.bit_array[result] == 0: 
				return "Nope" 
		return "Probably" 
bf = BloomFilter(500000, 7) 
bf.add("dantezhao") 
print (bf.lookup("dantezhao")) 
print (bf.lookup("yyj")) 
```



## LRU Cache(LRU-least recently used)

1. 缓存有两个要素

- 大小
- 替换策略

2. Hash Table + Double LinkedList
3. O(1)查询、O(1)修改、更新

```python
class LRUCache(object): 

	def __init__(self, capacity): 
		self.dic = collections.OrderedDict() 
		self.remain = capacity

	def get(self, key): 
		if key not in self.dic: 
			return -1 
		v = self.dic.pop(key) 
		self.dic[key] = v   # key as the newest one 
		return v 

	def put(self, key, value): 
		if key in self.dic: 
			self.dic.pop(key) 
		else: 
			if self.remain > 0: 
				self.remain -= 1 
			else:   # self.dic is full
				self.dic.popitem(last=False) 
		self.dic[key] = value
```



# 8-18 排序算法

**比较类排序，非比较类排序**

## 比较类排序

1. 交换排序

- 冒泡排序
- 快速排序

2. 插入排序

- 简单插入排序
- 希尔排序

3. 选择排序

- 简单选择排序
- 堆排序

4. 归并排序

- 二路归并排序
- 多路归并排序

## 非比较排序

1. 计数排序

1. 桶排序
2. 基数排序

### 冒泡排序

```python
def bubble_sort(nums):
    for i in range(len(nums) - 1):
        for j in range(len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
```

### 选择排序

```python
def selection_sort(nums):
    for i in range(len(nums) - 1):
        min_idx = i
        for j in range(i + 1, len(nums)):
            if nums[min_idx] > nums[j]:
                min_idx = j
        nums[i], nums[min_idx] = nums[min_idx], nums[i]
```

### 插入排序

```python
def insert_sort(nums):
    for i in range(len(nums) - 1):
        pre, cur = i, i + 1
        while nums[pre] > nums[cur] and pre >= 0:
            nums[pre + 1] = nums[pre]
            pre -= 1
        nums[pre] = nums[cur]        
```



### 快速排序

```python
def quick_sort(begin, end, nums):
    if begin >= end:
        return
    pivot_index = partition(begin, end, nums)
    quick_sort(begin, pivot_index-1, nums)
    quick_sort(pivot_index+1, end, nums)
    
def partition(begin, end, nums):
    pivot = nums[begin]
    mark = begin
    for i in range(begin+1, end+1):
        if nums[i] < pivot:
            mark +=1
            nums[mark], nums[i] = nums[i], nums[mark]
    nums[begin], nums[mark] = nums[mark], nums[begin]
    return mark
```

### 归并排序

```python
def mergesort(nums, left, right):
    if right <= left:
        return
    mid = (left+right) >> 1
    mergesort(nums, left, mid)
    mergesort(nums, mid+1, right)
    merge(nums, left, mid, right)

def merge(nums, left, mid, right):
    temp = []
    i = left
    j = mid+1
    while i <= mid and j <= right:
        if nums[i] <= nums[j]:
            temp.append(nums[i])
            i +=1
        else:
            temp.append(nums[j])
            j +=1
    while i<=mid:
        temp.append(nums[i])
        i +=1
    while j<=right:
        temp.append(nums[j])
        j +=1
    nums[left:right+1] = temp
```

### 堆排序

```python
def heapify(parent_index, length, nums):
    temp = nums[parent_index]
    child_index = 2*parent_index+1
    while child_index < length:
        if child_index+1 < length and nums[child_index+1] > nums[child_index]:
            child_index = child_index+1
        if temp > nums[child_index]:
            break
        nums[parent_index] = nums[child_index]
        parent_index = child_index
        child_index = 2*parent_index + 1
    nums[parent_index] = temp


def heapsort(nums):
    for i in range((len(nums)-2)//2, -1, -1):
        heapify(i, len(nums), nums)
    for j in range(len(nums)-1, 0, -1):
        nums[j], nums[0] = nums[0], nums[j]
        heapify(0, j, nums)
```





