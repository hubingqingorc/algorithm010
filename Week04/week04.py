## 1.柠檬水找零
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        count_5, count_10 = 0, 0
        for i in bills:
            if i == 5:
                count_5 += 1
            elif i == 10:
                if count_5 > 0:
                    count_5 -= 1
                else:
                    return False
                count_10 += 1
            else:
                if count_10 > 0 and count_5 > 0:
                    count_10 -= 1
                    count_5 -= 1
                elif count_5 > 3:
                        count_5 -= 3
                else:
                        return False
        return True

## 2.买卖股票的最佳时机 II
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        length = len(prices)
        if length < 2:
            return profit
        for i in range(length-1):
            if prices[i] < prices[i+1]:
                profit += prices[i+1] - prices[i]
        return profit

## 3.分发饼干
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        sort_g, sort_s = sorted(g), sorted(s)
        count = 0
        len_g, len_s = len(g), len(s)
        for i in range(len_s):
            if count == len_g:
                break
            if sort_s[i] >= sort_g[count]:
                count += 1
        return count

## 4.模拟行走机器人
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        obstacles = set(map(tuple, obstacles))
        direction = 1   # 0-west, 1-north, 2-east, 3-south
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
        x, y = 0, 0
        dist = 0
        for i in commands:
            if i == -2:
                direction = (direction - 1) % 4
            elif i == -1:
                direction = (direction + 1) % 4
            else:
                for _ in range(i):
                    if (x + dx[direction], y + dy[direction]) not in obstacles:
                        x += dx[direction]
                        y += dy[direction]
                dist = max(dist, x**2 + y**2)
        return dist

## 5.单词接龙
class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        """
        1.BFS
        """
        node = collections.defaultdict(list)
        for i in wordList:
            for j in range(len(i)):
                node[i[0:j] + "*" + i[j + 1:]].append(i)

        quene = collections.deque([beginWord])
        visit = {beginWord}
        count = 1
        while quene:
            count += 1
            for _ in range(len(quene)):
                beginWord = quene.popleft()
                for j in range(len(beginWord)):
                    linked_node = node[beginWord[0:j] + "*" + beginWord[j + 1:]]
                    for k in linked_node:
                        if k not in visit:
                            if k == endWord:
                                return count
                            visit.add(k)
                            quene.append(k)
        return 0

## 6.岛屿数量
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        1.dfs
        2.bfs
        """
        count = 0
        if len(grid) == 0 or len(grid[0]) == 0:
            return count

        len_x, len_y = len(grid), len(grid[0])
        def dfs(x, y):
            if x < 0 or x >= len_x or y < 0 or y >= len_y:
                return
            if grid[x][y] == '1':
                grid[x][y] = '0'
                dfs(x - 1, y)
                dfs(x + 1, y)
                dfs(x    , y - 1)
                dfs(x    , y + 1)
            return

        for i in range(len_x):
            for j in range(len_y):
                if grid[i][j] == '1':
                    count += 1
                    dfs(i, j)
        return count

## 7.扫雷游戏
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        """
        1.dfs
        2.bfs
        """
        x, y = click
        len_x, len_y = len(board), len(board[0])

        def dfs(x, y):
            if x < 0 or x >= len_x or y < 0 or y >= len_y:
                return
            if board[x][y] == 'M':  # 1.挖到雷上
                board[x][y] = 'X'
                return
            elif board[x][y] == 'E':
                num_mine = check(x, y)
                if num_mine:
                    board[x][y] = str(num_mine)
                else:
                    board[x][y] = 'B'
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            if i == 0 and j == 0:
                                continue
                            dfs(x + i, y + j)
                return
            return

        def check(x, y):
            num_mine = 0
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    x_1, y_1 = x + i, y + j
                    if x_1 < 0 or x_1 >= len_x or y_1 < 0 or y_1 >= len_y:
                        continue
                    if board[x_1][y_1] == 'M':
                        num_mine += 1
            return num_mine

        dfs(x, y)
        return board

## 8.跳跃游戏
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        length = len(nums)
        cur_pos, max_pos = 0, 0
        while cur_pos <= max_pos:
            max_pos = max(max_pos, cur_pos+nums[cur_pos])
            if max_pos >= length-1:
                return True
            cur_pos += 1
        return False

## 9.搜索旋转排序数组
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif (nums[left] < nums[mid] and nums[left] <= target < nums[mid]) or (nums[mid] < nums[right] and (target < nums[mid] or target > nums[right])):
                right = mid - 1
            else:
                left = mid + 1
        return -1

## 10.搜索二维矩阵
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        if m == 0:
            return False
        n = len(matrix[0])
        if n == 0:
            return False
        left, right = 0, m * n - 1
        while left <= right:
            mid = (left + right) // 2
            x, y = mid // n, mid % n
            if matrix[x][y] == target:
                return True
            elif target < matrix[x][y]:
                right = mid - 1
            else:
                left = mid + 1
        return False

## 11.单词接龙 II
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        wordList_set = set(wordList)
        res = []
        if endWord not in wordList_set or len(wordList_set) == 0:
            return res

        use_node = collections.defaultdict(set)
        if self.bfs(beginWord, endWord, wordList_set, use_node):
            ans = [beginWord]
            self.dfs(beginWord, endWord, use_node, ans, res)
        return res

    def bfs(self, beginWord, endWord, wordList_set, use_node):
        reach = False
        quene = collections.deque([beginWord])
        early_visit = {beginWord}
        len_word = len(beginWord)
        while quene:
            if reach:
                break
            now_visit = set()
            length = len(quene)
            for i in range(length):
                ans = quene.popleft()
                ans_list = list(ans)
                for j in range(len_word):
                    origin_char = ans_list[j]
                    for k in string.ascii_lowercase:
                        ans_list[j] = k
                        new_word = "".join(ans_list)
                        if new_word in wordList_set:
                            if new_word == endWord:
                                reach = True
                            if new_word not in early_visit:
                                use_node[ans].add(new_word)
                                quene.append(new_word)
                                if new_word not in now_visit:
                                    now_visit.add(new_word)
                    ans_list[j] = origin_char
            early_visit = early_visit | now_visit
        return reach

    def dfs(self, beginWord, endWord, use_node, ans, res):
        if ans[-1] == endWord:
            res.append(ans[:])
            return
        if beginWord not in use_node:
            return
        for i in use_node[ans[-1]]:
            ans.append(i)
            self.dfs(i, endWord, use_node, ans, res)
            ans.pop()

## 12.跳跃游戏 II
class Solution:
    def jump(self, nums: List[int]) -> int:
        length = len(nums)
        if length < 2:
            return 0
        cur_pos, max_pos = 0, 0
        count = 1
        while cur_pos <= max_pos:
            max_pos = max(max_pos, cur_pos + nums[cur_pos])
            if max_pos >= length - 1:
                return count
            max_step, max_idx = -1, -1
            right = max_pos + 1 if max_pos + 1 <= length else length
            for i in range(cur_pos + 1, right):
                if nums[i] + i >= max_step:
                    max_step = nums[i] + i
                    max_idx = i
            count += 1
            cur_pos = max_idx