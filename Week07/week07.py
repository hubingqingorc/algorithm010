# 1 https://leetcode-cn.com/problems/climbing-stairs/
class Solution:
    def climbStairs(self, n: int) -> int:
        # f(n) = f(n-1) + f(n-2) f(1) = 1 f(2) = 2
        if n < 2:
            return n
        f = [0] * n
        f[0], f[1] = 1, 2
        for i in range(2, n):
            f[i] = f[i-1] + f[i-2]
        return f[-1]

# 2 https://leetcode-cn.com/problems/implement-trie-prefix-tree/#/description
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node["#"] = None

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        if "#" not in node:
            return False
        return True

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

# 3 https://leetcode-cn.com/problems/friend-circles/
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        length = len(M)
        visited = [0] * length
        def dfs(i, visited, M):
            for j in range(length):
                if visited[j] == 0 and M[i][j] == 1:
                    visited[j] = 1
                    dfs(j, visited, M)
        num = 0
        for i in range(length):
            if visited[i] == 0:
                dfs(i, visited, M)
                num += 1
        return num

# 4 https://leetcode-cn.com/problems/number-of-islands/
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        if m == 0:
            return 0
        num = 0
        n = len(grid[0])
        disjoint = [i for i in range(m * n)]  # 并查集

        def find_root(i, j, disjoint, n):  # 找根结点
            root = i * n + j
            while disjoint[root] != root:
                root = disjoint[root]
            idx = i * n + j
            while disjoint[idx] != root:  # 压缩路径
                temp = disjoint[idx]
                disjoint[idx] = root
                idx = temp
            return root

        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    root = find_root(i, j, disjoint, n)  # 返回根结点
                    for dx, dy in [(1, 0), (0, 1)]:
                        x, y = i + dx, j + dy
                        if 0 <= x < m and 0 <= y < n and grid[x][y] == "1":
                            root_neibor = find_root(x, y, disjoint, n)
                            disjoint[root_neibor] = root  # 合并
        num = set()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":  # 仅统计陆地
                    root = find_root(i, j, disjoint, n)
                    num.add(root)
        return len(num)

# 5 https://leetcode-cn.com/problems/surrounded-regions/
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(board)
        if m == 0:
            return
        n = len(board[0])
        plus = m * n  # 设定虚拟代表
        disjoint = [i for i in range(m * n + 1)]  # 并查集尺寸增加1
        for i in [0, m - 1]:
            for j in range(n):
                if board[i][j] == "O":
                    disjoint[i * n + j] = plus  # 边界O指向虚拟代表
        for j in [0, n - 1]:
            for i in range(m):
                if board[i][j] == "O":
                    disjoint[i * n + j] = plus  # 边界O指向虚拟代表

        def find_root(i, j, disjoint, n):
            idx = i * n + j
            root = idx
            while disjoint[root] != root:
                root = disjoint[root]
            while disjoint[idx] != root:
                temp = idx
                idx = disjoint[idx]
                disjoint[temp] = root
            return root

        for i in range(m):
            for j in range(n):
                if board[i][j] == "O":
                    root = find_root(i, j, disjoint, n)
                    for dx, dy in [(1, 0), (0, 1)]:
                        x, y = i + dx, j + dy
                        if 0 <= x < m and 0 <= y < n and board[x][y] == "O":
                            root_neibor = find_root(x, y, disjoint, n)
                            if root == plus or root_neibor == plus:
                                disjoint[root] = plus
                                disjoint[root_neibor] = plus
                            else:
                                disjoint[root_neibor] = root
        for i in range(m):
            for j in range(n):
                if board[i][j] == "O":
                    if find_root(i, j, disjoint, n) != plus:
                        board[i][j] = "X"

# 6 https://leetcode-cn.com/problems/valid-sudoku/description/
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row, col, part = [set() for _ in range(9)], [set() for _ in range(9)], [set() for _ in range(9)]
        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    if board[i][j] in row[i]:
                        return False
                    else:
                        row[i].add(board[i][j])
                    if board[i][j] in col[j]:
                        return False
                    else:
                        col[j].add(board[i][j])
                    if board[i][j] in part[i//3 * 3 + j//3]:
                        return False
                    else:
                        part[i//3 * 3 + j//3].add(board[i][j])
        return True

# 7 https://leetcode-cn.com/problems/generate-parentheses/
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        def recursion(res, n, s, left, right):
            if left > n or right > left:
                return
            if left + right == n * 2:
                res.append(s)
                return
            else:
                recursion(res, n, s + "(", left + 1, right)
                recursion(res, n, s + ")", left, right + 1)
        recursion(res, n, "", 0, 0)
        return res

# 8 https://leetcode-cn.com/problems/word-ladder/
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0
        length = len(beginWord)
        edge = collections.defaultdict(list)
        for i in range(len(wordList)):
            for j in range(length):
                edge[wordList[i][:j] + "*" + wordList[i][j+1:]].append(wordList[i])
        quene1 = collections.deque([beginWord])
        quene2 = collections.deque([endWord])
        visit = {beginWord, endWord}
        level1, level2 = {beginWord}, {endWord}
        count_level = 2
        while len(quene1) != 0:
            for _ in range(len(quene1)):
                node = quene1.popleft()
                for i in range(length):
                    key = node[:i] + "*" + node[i+1:]
                    for j in edge[key]:
                        if j in level2:
                            return count_level
                        if j not in visit:
                            visit.add(j)
                            quene1.append(j)
            count_level += 1
            level1 = set(quene1)
            if len(quene1) > len(quene2):
                quene1, quene2 = quene2, quene1
                level1, level2 = level2, level1
        return 0

# 9 https://leetcode-cn.com/problems/minimum-genetic-mutation/
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        bank = set(bank)
        if end not in bank:
            return -1
        length = len(start)
        relate = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}
        set_1, set_2 = {start}, {end}
        count = 1
        while len(set_1) != 0:
            new_set = set()
            for i in set_1:
                for j, c in enumerate(i):
                    for k in relate[c]:
                        new = i[:j] + k + i[j+1:]
                        if new in set_2:
                            return count
                        if new in bank:
                            new_set.add(new)
                            bank.remove(new)
            count += 1
            set_1 = new_set
            if len(set_1) > len(set_2):
                set_1, set_2 = set_2, set_1
        return -1

# 10 https://leetcode-cn.com/problems/word-search-ii/
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = {}
        for s in words:
            node = root
            for c in s:
                if c not in node:
                    node[c] = {}
                node = node[c]
            node["#"] = None

        def dfs(i, j, s, parent, board, res):
            if board[i][j] != "@" and board[i][j] in parent:
                letter = board[i][j]
                s += letter
                node = parent[letter]
                board[i][j] = "@"
                if "#" in node:
                    res.append(s)
                    node.pop("#")
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    x, y = i + dx, j + dy
                    if 0 <= x < m and 0 <= y < n:
                        dfs(x, y, s, node, board, res)
                board[i][j] = letter
                if len(node) == 0:
                    parent.pop(letter)

        m, n = len(board), len(board[0])
        res = []
        for i in range(m):
            for j in range(n):
                dfs(i, j, "", root, board, res)
        return res

# 11 https://leetcode-cn.com/problems/n-queens/
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        shu, pie, na = [False] * n, [False] * (n * 2 - 1), [False] * (n * 2 - 1)
        res, queen_pos = [], []
        def recursion():
            row = len(queen_pos)
            if row == n:
                res.append(queen_pos[:])
                return
            for col in range(n):
                if shu[col] == False and pie[row+col] == False and na[n-1-row+col] == False:
                        shu[col], pie[row + col], na[n - 1 - row + col] = True, True, True
                        queen_pos.append(col)
                        recursion()
                        shu[col], pie[row + col], na[n - 1 - row + col] = False, False, False
                        queen_pos.pop()
        recursion()
        output = [["." * i + "Q" + "." * (n-i-1) for i in j] for j in res]
        return output

# 12 https://leetcode-cn.com/problems/sudoku-solver/#/description
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row, col, part = [set() for _ in range(9)], [set() for _ in range(9)], [set() for _ in range(9)]
        blank = []
        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    row[i].add(board[i][j])
                    col[j].add(board[i][j])
                    part[i//3 * 3 + j//3].add(board[i][j])
                else:
                    blank.append([i, j])
        def recursion(row, col, part, blank, board, count, n):
            if count == n:
                return True
            else:
                x, y = blank.pop()
                for c in range(1, 10):
                    c = str(c)
                    if c not in row[x] and c not in col[y] and c not in part[x//3 * 3 + y//3]:
                        row[x].add(c)
                        col[y].add(c)
                        part[x//3 * 3 + y//3].add(c)
                        board[x][y] = c
                        count += 1
                        check = recursion(row, col, part, blank, board, count, n)
                        if check:
                            return check
                        row[x].remove(c)
                        col[y].remove(c)
                        part[x//3 * 3 + y//3].remove(c)
                        board[x][y] = "."
                        count -= 1
                blank.append([x,y])
                return False
        count, n = 0, len(blank)
        recursion(row, col, part, blank, board, count, n)