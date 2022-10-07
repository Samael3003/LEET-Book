# DAY - I


### 896. Monotonic Array
class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        
        countf, countb, nums_len = 0, 0, len(nums)-1
        
        for i in range(nums_len):
            cur, nxt = nums[i], nums[i+1]
            if nxt >= cur: 
                countf += 1
            if nxt <= cur: 
                countb += 1
        
        if countf == nums_len or countb == nums_len: 
            return True
        return False
        

### 28. Find the Index of the First Occurrence in a String
        class Solution:
            def strStr(self, haystack: str, needle: str) -> int:
                s1=needle[0]
                n=len(needle)
                for i in range(len(haystack)):
                    if haystack[i]==s1:
                        if haystack[i:i+n]==needle:
                            return i
                return -1        



# DAY - II

### 110. Balanced Binary Tree
        class Solution:
            def isBalanced(self, root: Optional[TreeNode]) -> bool:
                ans = True

                def rec(node, height):
                    nonlocal ans
                    if node is None or not ans:
                        return height
                    left_height = rec(node.left, height + 1)
                    right_height = rec(node.right, height + 1)
                    if abs(left_height - right_height) > 1:
                        ans = False
                    return max(left_height, right_height)

                rec(root, 0)
                return ans

### 459. Repeated Substring Pattern
        class Solution:
            def repeatedSubstringPattern(self, s: str) -> bool:
                d = [int(len(s) ** 0.5)] if int(len(s) ** 0.5) ** 2 == len(s) else []
                for i in range(1, math.ceil(len(s) ** 0.5)):
                    if len(s) % i == 0: d.extend([i, int(len(s) / i)])
                d = list(filter((len(s)).__ne__, d))
                # make sure to remove all len(s) to accomodate to len(s) == 1
                for j in set(d):
                    if s[:j] * int(len(s) / j) == s: return True
                return False


# DAY - III

### 66. Plus One
        class Solution:
            def plusOne(self, digits: List[int]) -> List[int]:
                les="".join(list(map(lambda x:str(x),digits)))
                i=int(les)+1
                res=list(map(lambda x:int(x),list(str(i))))
                return res



### 150. Evaluate Reverse Polish Notation
        class Solution:
            def evalRPN(self, tokens: List[str]) -> int:


                stack = []
                for c in tokens:
                    if c == "+":
                        val = stack.append(stack.pop() + stack.pop())
                    elif c == "-":

                        a,b =stack.pop(),stack.pop()
                        stack.append(b-a)

                    elif c == "*":
                        val = stack.append(stack.pop() * stack.pop())

                    elif c == "/":
                        a,b =stack.pop(),stack.pop()
                        stack.append(int(b/a))

                    else:
                        stack.append(int(c))

                return stack[0]


# DAY - IV

### 1367. Linked List in Binary Tree
        # IN JAVASCRIPT
        var isSubPath = function(head, root) {
            if(!root)    return false
            if(issame(head, root))  return true
            return isSubPath(head, root.left) || isSubPath(head, root.right)
        };

        function issame(head, root){
            if(!head)   return true
            if(!root)   return false
            if(head.val != root.val)    return false
            return issame(head.next, root.left) || issame(head.next, root.right)
        };


### 43. Multiply Strings
        class Solution:
            def multiply(self, num1: str, num2: str) -> str:
                return str(int(num1)*int(num2))
