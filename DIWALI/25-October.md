

## 92. Reverse Linked List II

    class Solution {
        public ListNode reverseBetween(ListNode head, int left, int right) {
            ListNode dummy = new ListNode(0); // created dummy node
            dummy.next = head;
            ListNode prev = dummy; // intialising prev pointer on dummy node

            for(int i = 0; i < left - 1; i++)
                prev = prev.next; // adjusting the prev pointer on it's actual index

            ListNode curr = prev.next; // curr pointer will be just after prev
            // reversing
            for(int i = 0; i < right - left; i++){
                ListNode forw = curr.next; // forw pointer will be after curr
                curr.next = forw.next;
                forw.next = prev.next;
                prev.next = forw;
            }
            return dummy.next;
        }
    }
    
> Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Linked List II.

> Memory Usage: 40.9 MB, less than 83.65% of Java online submissions for Reverse Linked List II.





## 86. Partition List

        class Solution {
            public ListNode partition(ListNode head, int x) {
                ListNode fdum = new ListNode(0), bdum = new ListNode(0),
                         front = fdum, back = bdum, curr = head;
                while (curr != null) {
                    if (curr.val < x) {
                        front.next = curr;
                        front = curr;
                    } else {
                        back.next = curr;
                        back = curr;
                    }
                    curr = curr.next;
                }
                front.next = bdum.next;
                back.next = null;
                return fdum.next;
            }
        }

> Runtime: 1 ms, faster than 79.28% of Java online submissions for Partition List.

> Memory Usage: 42.5 MB, less than 51.21% of Java online submissions for Partition List.





## 315. Count of Smaller Numbers After Self

    class Solution {    
        public List<Integer> countSmaller(int[] nums) {
            int min = 20001;
            int max = -1;
            for (int num : nums) {
                min = Math.min(min, num);
                max = Math.max(max, num);
            }

            min--;
            int[] count = new int[max-min+1];
            Integer[] result = new Integer[nums.length];
            for (int i = nums.length-1; i >=0; i--) {
                int k = nums[i]-min-1;
                int c = 0;
                do {
                    c += count[k];
                    k -= (-k&k);
                } while (k > 0);
                result[i] = c;

                k = nums[i]-min;
                while (k < count.length) {
                    count[k]++;
                    k += (-k&k);
                }
            }

            return Arrays.asList(result);
        }
    }
    
> Runtime: 27 ms, faster than 98.88% of Java online submissions for Count of Smaller Numbers After Self.

> Memory Usage: 127.7 MB, less than 82.50% of Java online submissions for Count of Smaller Numbers After Self.
    



## 240. Search a 2D Matrix II

    class Solution {
        public boolean searchMatrix(int[][] matrix, int target) {
            int rowLength = matrix.length;
            int colLength = matrix[0].length;
            int start =0;
            int end = colLength-1;
            while(start < rowLength && end >= 0){

                if(matrix[start][end] == target){
                    return true;
                }
                else if(matrix[start][end] > target){
                    end--;
                }
                else {
                    start++;
                }
            }
            return false;
        }
    }

> Runtime: 14 ms, faster than 28.67% of Java online submissions for Search a 2D Matrix II.

> Memory Usage: 52.9 MB, less than 84.12% of Java online submissions for Search a 2D Matrix II.





## 34. Find First and Last Position of Element in Sorted Array

    class Solution {
        public int[] searchRange(int[] nums, int target) {
            int []arr = {-1, -1};
            int i;
            for(i=0; i<nums.length; i++){
                if(nums[i] == target){
                    arr[0] = i;
                    break;
                }
            }
            if(arr[0] == -1)
                return arr;
            while(i < nums.length && nums[i] == target)
                ++i;
            arr[1] = --i;

            return arr;
        }
    }
    
> Runtime: 3 ms, faster than 8.11% of Java online submissions for Find First and Last Position of Element in Sorted Array.
> Memory Usage: 47.1 MB, less than 57.55% of Java online submissions for Find First and Last Position of Element in Sorted Array.    





## 236. Lowest Common Ancestor of a Binary Tree


    class Solution {
        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            if (root == null || root == p || root == q) return root;
            TreeNode left = lowestCommonAncestor(root.left, p, q);
            TreeNode right = lowestCommonAncestor(root.right, p, q);
            return left == null ? right : right == null ? left : root;
        }
    }

> Runtime: 7 ms, faster than 89.87% of Java online submissions for Lowest Common Ancestor of a Binary Tree.

> Memory Usage: 43.2 MB, less than 97.68% of Java online submissions for Lowest Common Ancestor of a Binary Tree.





## 242. Valid Anagram

    class Solution {
    public boolean isAnagram(String s, String t)
    {
    char arr[] = s.toCharArray();
    char brr[] = t.toCharArray();

        Arrays.sort(arr);
        Arrays.sort(brr);

        if(Arrays.equals(arr,brr))
        {
            return true;
        }
        else 
        {
            return false;
        }
    }
    }

> Runtime: 3 ms, faster than 96.35% of Java online submissions for Valid Anagram.

> Memory Usage: 42.2 MB, less than 94.87% of Java online submissions for Valid Anagram.
