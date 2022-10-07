# Day - I

### 1706. Where Will the Ball Fall
        class Solution(object):
            def findBall(self, grid):
                rows = len(grid)
                cols = len(grid[0])
                ans = []
                for ball in range(cols):
                    for row in grid:
                        if (  # stuck
                            (ball == 0 and row[0] == -1)  # at left edge
                            or (ball == cols - 1 and row[-1] == 1)  # at right edge
                            or (row[ball] == 1 and row[ball+1] == -1)  # "\/" on left
                            or (row[ball] == -1 and row[ball-1] == 1)  # "\/" on right
                        ):
                            ball = -1
                            break
                        ball += row[ball]
                    ans.append(ball)
                return ans

        
### 202. Happy Number        
        class Solution(object):
            def isHappy(self, n):
                seen = set() # to store all the sum of square if digits
                while n != 1:
                    sums = sum([int(digit)**2 for digit in str(n)]) # sum of square of digits
                    if sums in seen: # if the sums in present in seen then it will be a endless loop
                        return False
                    n = sums
                    seen.add(n)

                return True


### 54. Spiral Matrix
        class Solution:
            def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
                return matrix and [*matrix.pop(0)]+self.spiralOrder([*zip(*matrix)][::-1])



# Day - II

### 14. Longest Common Prefix
        class Solution:
            def longestCommonPrefix(self, strs: List[str]) -> str:
                # handling edge case, when there is only one string
                # the string itself is the common prefix
                if len(strs) == 1:
                    return strs[0]

                last_common = 0
                # iterate for each character in the first string
                for i in range(len(strs[0])):
                    last_common = i
                    curr_c = strs[0][i]

                    # iterate each string on a given character position 
                    for s in strs:
                        # if string is shorter than other string
                        # or string have different character
                                        # common prefix is the beginning until last character checked
                        if i >= len(s) or s[i] != curr_c:
                            return s[:last_common]

                # if all passed than return the whole first string 
                return strs[0]

        #     Time complexity is O(n + m), where n is the number of string in list and m is the shortest string length
        #     Space complexity is O(1) as the code doesn't grow in space with growing input
        
    
### 43. Multiply Strings 
        class Solution:
            def multiply(self, num1: str, num2: str) -> str:
                return str(int(num1)*int(num2))
                
                
                
                
# DAY - III

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
        

### 234. Palindrome Linked List
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def isPalindrome(self, head: Optional[ListNode]) -> bool:
                li = []
                while head is not None:
                    li.append(head.val)
                    head = head.next

                j = len(li)-1
                i = 0
                while i < len(li)//2:
                    if li[i] != li[j]:
                        return False
                    i+=1
                    j-=1

                return True

# DAY - IV

### 328. Odd Even Linked List
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:

                #HERE WE TOOK TWO POINTERS ONE POINTING TO ODD PLACES AND THE OTHER TO EVEN PLACES

                if head == None: #CORNER CASE OF NO VALUE IN LL
                    return head


                odd = head
                even = head.next
                head2 = even 


                while even != None  and even.next != None:
                    odd.next = odd.next.next
                    even.next = even.next.next
                    odd = odd.next
                    even = even.next

                odd.next = head2


                return head


### 148. Sort List
        class Solution:
            def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
                if head is None or head.next is None:
                    return head

                def getMid(head):
                    slow, fast = head, head.next
                    while fast and fast.next:
                        slow = slow.next
                        fast = fast.next.next
                    return slow

                def merge(list1, list2):
                    curr = sentinel = ListNode()
                    while list1 and list2:
                        if list1.val < list2.val:
                            curr.next = list1
                            list1 = list1.next
                        else:
                            curr.next = list2
                            list2 = list2.next
                        curr = curr.next
                    if list1:
                        curr.next = list1
                    if list2:
                        curr.next = list2
                    return sentinel.next

                left = head
                right = getMid(head)
                tmp = right.next
                right.next = None
                right = tmp
                left = self.sortList(left)
                right = self.sortList(right)
                return merge(left, right)
            # Time: O(nlog(n))
            # Space: O(log(n))

