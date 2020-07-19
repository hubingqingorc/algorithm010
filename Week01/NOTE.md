# 1-3 数组 链表 跳表

## 数组 Array

查询 O(1)

队尾增加、删除 O(1)

删除中间 O(n)

## 链表

```python
class LinkedListNode:
    def __init__(self, val):
		self.val = val
        self.next = None  
```

查询O(n)

增加、删除 O(1)

**双向链表**

```python
class DLNode:
    def __init__(self, val):
        self.val = val
        self.left_node = None
        self.right_node = None
```

**跳表**

对应元素有序的情况

对标平衡数（AVL Tree）、二分查找，插入、删除、搜索均为log(n)

优势：原理简单、容易实现、方便扩展、效率更高，一些热门项目中用来替代平衡树，如Redis、LevelDB等。

空间复杂度 O(n)

跳表体现升维思想->空间换时间



# 习题

1. 盛水最多的容器->双指针

2. 移动零->快慢指针

3. 爬楼梯->斐波那契数列，数学，转移矩阵
4. sum3
5. 环形链表
6. 反转链表
7. 逐个交换两个链表
8. 环形链表1、环形链表2

# 1-4 栈、队列、优先队列、双端队列

# 栈、队列特效

1. 栈 后进先出/先进后出 添加删除均为 O(1)
2. 队列 先进 先出 添加删除均为 O(1)
3. 双端队列 添加、删除均为O(1)
4. 优先队列，插入O(1)，取出O(logN)，实现方式堆(heap)/二叉搜索树(binary search tree,-bst)/平衡二叉树Treap = Tree + heap
5. ![image-20200710211733898](C:\Users\HBQ\AppData\Roaming\Typora\typora-user-images\image-20200710211733898.png)

## 习题

1. 最小栈
2. 有效括号
3. 滑动窗口
4. 