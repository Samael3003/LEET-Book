# DAY-I


### 704. Binary Search
        class Solution:
            def search(self, nums: List[int], target: int) -> int:
                start = 0
                end = len(nums)
                for value in range(len(nums) // 2 +1 )   :
                    median = ((end - start) // 2) + start
                    if nums[median] == target:
                        return median
                    elif nums[median] > target:
                        end = median
                    elif nums[median] < target:
                        start = median
                return -1



### 278. First Bad Version
        class Solution:
            def firstBadVersion(self, n: int) -> int:
                if n == 1:
                    return 1
                    
                low, high = 1, n

                while(low <= high):
                    mid = low + (high - low) // 2

                    if isBadVersion(mid):
                        high = mid - 1

                    else:
                        low = mid + 1

                return low
                
                
### 35. Search Insert Position 
        class Solution(object):
            def searchInsert(self, nums, target):
                nums.append(target)
                nums.sort()
                return nums.index(target)               


# DAY - II

### 977. Squares of a Sorted Array
        class Solution:
            def sortedSquares(self, nums: List[int]) -> List[int]:
                arr=[i*i for i in nums]
                arr.sort()
                return arr
                
### 189. Rotate Array
        class Solution(object):
            def rotate(self, nums, k):
                """
                :type nums: List[int]
                :type k: int
                :rtype: None Do not return anything, modify nums in-place instead.
                """
                n = len(nums)
                k = k % n
                self.reverse(nums, 0, n - 1)
                self.reverse(nums, 0, k - 1)
                self.reverse(nums, k, n - 1)

            def reverse(self, nums, start, end):
                while start < end:
                    nums[start], nums[end] = nums[end], nums[start]
                    start += 1
                    end -= 1


# DAY - III

### 167. Two Sum II - Input Array Is Sorted
        class Solution:
            def twoSum(self, numbers: List[int], target: int) -> List[int]:
                low=0 #Start Index
                high=len(numbers)-1 #nth Index
                while(low<high):
                    if(numbers[low]+numbers[high]>target):
                        high-=1
                    elif(numbers[low]+numbers[high]<target):
                        low+=1
                    else:
                        return [low+1,high+1] 
                        #Adding +1, since in problem - Array index starts from 1


### 283. Move Zeroes
        class Solution(object):
            def moveZeroes(self, nums):
                l = 0 #lefft array
                for r in range(len(nums)): #right array
                    if nums[r]:
                        nums[l], nums[r] = nums[r], nums[l]  # Swap the left and rights
                        l+=1 
                return nums


# DAY - IV
### 344. Reverse String
        class Solution:
            def reverseString(self, s: List[str]) -> None:
                """
                Do not return anything, modify s in-place instead.
                """
                l = 0;
                r = len(s) -1;
                while(l < r):
                    temp = s[l];
                    s[l] = s[r];
                    s[r] = temp;
                    l = l +1;
                    r = r -1;        

### 557. Reverse Words in a String III          
        class Solution:
            def reverseWords(self, s: str) -> str:

                news = s.split()

                save = []
                for i in news:
                    save.append(i[::-1])

                return ' '.join(save)



# DAY - V

### 876. Middle of the Linked List
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
                # find length of the linked list
                right=head
                l = 1
                while(right.next is not None):
                    right = right.next
                    l += 1
                # above in l we have length

                # now increment till half of length
                for i in range(l//2):
                    head = head.next
                return head


### 19. Remove Nth Node From End of List
        # Definition for singly-linked list.
        # class ListNode(object):
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution(object):
            def removeNthFromEnd(self, head, n):
                """
                :type head: ListNode
                :type n: int
                :rtype: ListNode
                """
                i = head
                length = 0
                while i:
                    i = i.next
                    length += 1

                if length == n:
                    return head.next
                j = head
                for x in range(length-n-1):
                    j = j.next
                j.next = j.next.next
                return head


# DAY - VI

### 3. Longest Substring Without Repeating Characters
        class Solution:
            def lengthOfLongestSubstring(self, s: str) -> int:
                charset=set()
                l=0
                result=0
                for r in range(len(s)):
                    while s[r] in charset:
                        charset.remove(s[l])
                        l+=1
                    charset.add(s[r])
                    result=max(result,r-l+1)
                return result

### 567. Permutation in String
        class Solution(object):
            def checkInclusion(self, s1, s2):
                """
                :type s1: str
                :type s2: str
                :rtype: bool
                """
                cntr, w, match = Counter(s1), len(s1), 0     

                for i in range(len(s2)):
                        if s2[i] in cntr:
                                if not cntr[s2[i]]: match -= 1
                                cntr[s2[i]] -= 1
                                if not cntr[s2[i]]: match += 1

                        if i >= w and s2[i-w] in cntr:
                                if not cntr[s2[i-w]]: match -= 1
                                cntr[s2[i-w]] += 1
                                if not cntr[s2[i-w]]: match += 1

                        if match == len(cntr):
                                return True

                return False



# DAY - VII

### 733. Flood Fill
        class Solution:
            def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
                if image[sr][sc]==color:
                    return image
                set_col=image[sr][sc]
                r,c=len(image),len(image[0])
                st=[(sr,sc)]
                direction=[(1,0),(-1,0),(0,1),(0,-1)]
                while len(st)>0:
                    temr,temc=st.pop(-1)
                    image[temr][temc]=color
                    for dr,dc in direction:
                        calr=temr+dr
                        calc=temc+dc
                        if 0<=calr<r and 0<=calc<c and image[calr][calc]==set_col:
                            st.append((calr,calc))
                return image


### 733. Flood Fill
        > IN JAVASCRIPT
        var maxAreaOfIsland = function(grid) {
                let result = 0;
                const M = grid.length;
                const N = grid[0].length;
                const isOutGrid = (m, n) => m < 0 || m >= M || n < 0 || n >= N;
                const island = (m, n) => grid[m][n] === 1;
                const dfs = (m, n) => {
                        if (isOutGrid(m, n) || !island(m, n)) return 0;

                        grid[m][n] = 'X';
                        const top = dfs(m - 1, n);
                        const bottom = dfs(m + 1, n);
                        const left = dfs(m, n - 1);
                        const right = dfs(m, n + 1);
                        return 1 + top + bottom + left + right;
                };

                for (let m = 0; m < M; m++) {
                        for (let n = 0; n < N; n++) {
                                if (!island(m, n)) continue;
                                const area = dfs(m, n);

                                result = Math.max(area, result);
                        }
                }
                return result;
        };        


# DAY - VIII

### 617. Merge Two Binary Trees
        class Solution(object):
            def mergeTrees(self, root1, root2):
                if not root1 and not root2:
                    return None 
                if root1 and not root2:
                    return root1
                if root2 and not root1:
                    return root2 

                node = TreeNode(root1.val+root2.val)
                node.left = self.mergeTrees(root1.left, root2.left )
                node.right = self.mergeTrees(root1.right, root2.right)
                return node 



### 116. Populating Next Right Pointers in Each Node
        class Solution:
            def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
                if not root : return None      #part1
                def recursion(cur):
                    if cur.left:                   #part2
                        cur.left.next = cur.right

                    if cur.next and cur.left:     #part3
                        cur.right.next = cur.next.left

                    recursion(cur.left) if cur.left else None        #part4
                    recursion(cur.right) if cur.right else None

                recursion(root)
                return root        

# DAY - IX

### 542. 01 Matrix
        class Solution:
            def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
                ROWS, COLS = len(mat), len(mat[0])
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                q = deque()
                visited = set()
                for row in range(ROWS):
                    for col in range(COLS):
                        if mat[row][col] == 0:
                            q.append((row, col, 0))
                            visited.add((row, col))
                res = [[0 if mat[row][col] == 0 else float('inf') for col in range(COLS)] for row in range(ROWS)]
                while q:
                    row, col, distance = q.popleft()
                    res[row][col] = min(res[row][col], distance)

                    for dRow, dCol in directions:
                        newRow, newCol = row + dRow, col + dCol
                        if (0 <= newRow < ROWS and
                            0 <= newCol < COLS and
                            (newRow, newCol) not in visited and
                            mat[newRow][newCol] == 1):
                            q.append((newRow, newCol, distance + 1))
                            visited.add((newRow, newCol))
                return res
            # Time: O(n) where n is the number of cells in mat
            # Space: O(n) it's the space for the queue and the visited set


### 994. Rotting Oranges
        class Solution:
            def orangesRotting(self, grid: List[List[int]]) -> int:
                visit = set()
                rows, cols = len(grid),len(grid[0])
                q = collections.deque()
                time , fresh = 0 , 0

                for r in range(rows):
                        for c in range(cols):
                            if grid[r][c] == 1 :
                                fresh += 1
                            if grid[r][c] == 2 :
                                q.append([r,c])

                while q and fresh > 0 :
                    for i in range(len(q)):
                        r,c = q.popleft()

                        directions = [[0,1], [1,0], [-1,0], [0,-1]]
                        for dr , dc in directions :
                            row , col = dr + r , dc + c

                            if (row < 0  or row == rows or col < 0 or col == cols or 
                            grid[row][col] != 1):
                                continue

                            grid[row][col] = 2
                            q.append([row,col])
                            fresh -= 1
                    time += 1
                return time if fresh == 0 else -1


# DAY - X


### 21. Merge Two Sorted Lists
        class Solution:
            def mergeTwoLists(self, l1 : ListNode, l2 : ListNode)-> ListNode:
                a = ListNode()
                b = a

                while l1 and l2:
                    if l1.val < l2.val:
                        b.next = l1
                        l1 = l1.next
                    else:
                        b.next = l2
                        l2 = l2.next
                    b = b.next

                if l1:
                    b.next = l1
                elif l2:
                    b.next = l2

                return a.next  

### 206. Reverse Linked List
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
                if not head:
                    return None
                prev = None

                while head:
                    next = head.next
                    head.next = prev

                    prev = head
                    head = next 
                return prev       

# DAY - XI

### 46. Permutations
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n=len(nums)
        i=0
        if n==1:
            return [nums]
        permut=[]
        while i<n:
        
        
            nums[0],nums[i]=nums[i],nums[0]
            for p in self.permute(nums[1:]):
                p.append(nums[0])
                permut.append(p)
            i+=1
        return permut


### 784. Letter Case Permutation
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
      result = [""]
      
      for char in s:
        if char.isalpha():
          #We need to permutate with the upper and lower cases
          result = [i + j for i in result for j in [char.upper(), char.lower()]]
        else:
          result = [i + char for i in result]
      
      return result


# DAY - XII

### 70. Climbing Stairs
        class Solution:
            def climbStairs(self, n: int) -> int:
                if(n<=2):
                    return n
                l=1
                r=2
                curr=0
                for i in range(2,n):
                    curr=l+r
                    l=r
                    r=curr
                return curr
        
        
### 198. House Robber
        class Solution:
            def rob(self, nums: List[int]) -> int:
                if len(nums) == 1:
                    return nums[0]
                first, second = nums[0], max(nums[0], nums[1])
                res = second
                for i in range(2, len(nums)):
                    first, second = second, max(nums[i] + first, second)
                    res = max(res, second)
                return res
            # Time: O(n)
            # Space: O(1)



### 120. Triangle
        class Solution:
            def minimumTotal(self, triangle: List[List[int]]) -> int:
                f = triangle[-1]
                for i in range(len(triangle) - 2, -1, -1):
                    for j in range(i + 1):
                        f[j] = min(f[j + 1], f[j]) + triangle[i][j]        
                return f[0]


# DAY - XIII

### 231. Power of Two        
        class Solution:
            def isPowerOfTwo(self, n: int) -> bool:
                while n>1:
                    n = n/2
                return n == 1


### 191. Number of 1 Bits
        class Solution:
            def hammingWeight(self, n: int) -> int:
                count = 0

                while n:
                    count += n & 1
                    n >>= 1

                return count


# DAY - XIV

### 190. Reverse Bits
        class Solution:
            def reverseBits(self, n: int) -> int:
                num, n1 = "0b",bin(n)
                numOfZeros = 34-len(n1)
                for idx in range(len(n1)-1,1,-1):
                    num += n1[idx]
                for _ in range(numOfZeros):
                    num += "0"
                return int(num,2)


### 136. Single Number
        class Solution(object):
            def singleNumber(self, nums):
                res = 0
                for i in nums:
                    res ^= i
                return res        
