# 001. Two Sum

    public class Solution {
        // example in leetcode book
        public int[] twoSum(int[] nums, int target) {
            Map<Integer, Integer> map = new HashMap<>();
            for (int i = 0; i < nums.length; i++) {
                int x = nums[i];
                if (map.containsKey(target - x)) {
                    return new int[]{map.get(target - x), i};
                }
                map.put(x, i);
            }
            throw new IllegalArgumentException("No two sum solution");
        }
    }


# 002. Add Two Numbers

    public class Solution {
        // example in leetcode book
        public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
          ListNode dummyHead = new ListNode(0);
          ListNode p = l1, q= l2, curr = dummyHead;
          int carry = 0;
          while (p != null || q!= null) {
            int x = (p != null) ? p.val : 0;
            int y = (q != null) ? q.val : 0;
            int digit = carry + x + y;
            carry = digit / 10;
            curr.next = new ListNode(digit % 10);
            curr = curr.next;
            if (p != null) p = p.next;
            if (q != null) q = q.next;
          }
          if (carry > 0) {
            curr.next = new ListNode(carry);
          }
          return dummyHead.next;
        }
    }


# 003. Longest Substring without Repeting

    public class Solution {
        public int lengthOfLongestSubstring(String s) {
          int[] charMap = new int[256];
          Arrays.fill(charMap, -1);
          int i = 0, maxLen = 0;
          for (int j = 0; j < s.length(); j++) {
            if (charMap[s.charAt(j)] >= i) {
              i = charMap[s.charAt(j)] + 1;
            }
            charMap[s.charAt(j)] = j;
            maxLen = Math.max(j - i + 1, maxLen);
          }
          return maxLen;
        }
    }


# 004. Median of Two Sorted Arrays

    public class Solution {
        // example in leetcode book
        public double findMedianSortedArrays(int[] nums1, int[] nums2) {
          int p1 = 0, p2 = 0, pos = 0;
          int ls1 = nums1.length, ls2 = nums2.length;
          int[] all_nums = new int[ls1+ls2];
          double median = 0.0;
          while (p1 < ls1 && p2 < ls2){
            if (nums1[p1] <= nums2[p2])
              all_nums[pos++] = nums1[p1++];
            else
              all_nums[pos++] = nums2[p2++];
          }
          while (p1 < ls1)
            all_nums[pos++] = nums1[p1++];
          while (p2 < ls2)
            all_nums[pos++] = nums2[p2++];
          if ((ls1 + ls2) % 2 == 1)
            median = all_nums[(ls1 + ls2) / 2];
          else
            median = (all_nums[(ls1 + ls2) / 2] + all_nums[(ls1 + ls2) / 2 - 1]) / 2.0;
            return median;
        }
    }


# 005. Longest Palindromic Substring

    public class Solution {
        // example in leetcode book
        public String longestPalindrome(String s) {
            int start = 0, end = 0;
            for (int i = 0; i < s.length(); i++) {
                // aba
                int len1 = expandAroundCenter(s, i, i);
                // bb
                int len2 = expandAroundCenter(s, i, i + 1);
                int len = Math.max(len1, len2);
                if (len > end - start) {
                    start = i - (len - 1) / 2;
                    end = i + len / 2;
                }
            }
            return s.substring(start, end + 1);
        }

        private int expandAroundCenter(String s, int left, int right) {
            int L = left, R = right;
            while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
                L--;
                R++;
            }
            return R - L - 1;
        }


# 007. Reverse Integer

    class Solution {
        public int reverse(int x) {
            if (x == 0) return 0;
            long res = 0;
            while (x != 0) {
                res = res * 10 + x % 10;
                if (res > Integer.MAX_VALUE || res < Integer.MIN_VALUE)
                    return 0;
                x /= 10;
            }
            return (int) res;
        }
    }


# 008. String to Integer

    public class Solution {
      // example in leetcode book
      private static final int maxDiv10 = Integer.MAX_VALUE / 10;
        public int myAtoi(String str) {
          int i = 0, n = str.length();
        while (i < n && Character.isWhitespace(str.charAt(i)))
          i++;
        int sign = 1;
        if (i < n && str.charAt(i) == '+')
          i++;
        else if (i < n && str.charAt(i) == '-') {
          sign = -1;
          i++;
        }
        int num = 0;
        while (i < n && Character.isDigit(str.charAt(i))) {
          int digit = Character.getNumericValue(str.charAt(i));
          if (num > maxDiv10 || num == maxDiv10 && digit >= 8)
            return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
          num = num * 10 + digit;
          i++;
        }
        return sign * num;    
        }
    }
    
    
# 009. Palindrome Number

    class Solution {
        public boolean isPalindrome(int x) {
            if (x < 0) 
                return false;
            int temp = x;
            int len = 0;
            while (temp != 0) {
                temp /= 10;
                len ++;
            }
            temp = x;
            int left, right;
            for (int i = 0; i < len / 2; i++) {
                right = temp % 10;
                left = temp / (int) Math.pow(10, len - 2 * i - 1);
                left = left % 10;
                if (left != right)
                    return  false;
                temp /= 10;
            }
            return true;
        }

# 011 Container with Most Water

    class Solution {
      public int maxArea(int[] height) {

        int maxArea = 0;
        int left = 0;
        int right = height.length - 1;

        while (left < right) {
          maxArea = Math.max(maxArea, (right - left) * Math.min(height[left], height[right]));
          // Two points
          if (height[left] < height[right]) left++;
          else right--;
        }
        return maxArea;
      }
    }


# 012. Iteger to Roman

    class Solution {
      public String intToRoman(int num) {
        Map<Integer, String> map = new HashMap();
        map.put(1, "I"); map.put(5, "V"); map.put(10, "X");
        map.put(50, "L"); map.put(100, "C"); map.put(500, "D"); map.put(1000, "M");
        map.put(4, "IV"); map.put(9, "IX"); map.put(40, "XL"); map.put(90, "XC");
        map.put(400, "CD"); map.put(900, "CM");

        int[] sequence = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};

        StringBuffer sb = new StringBuffer();
        for (int i = 0; i<sequence.length; i++) {
          int base = sequence[i];

          while (num >= base) {
            sb.append(map.get(base));
            num -= base;
          }
        }

        return sb.toString();
      }
    }

# 013. Roman to Integer

    class Solution {
        public int romanToInt(String s) {
            int[] arr = new int['A' + 26];
            arr['I'] = 1;
            arr['V'] = 5;
            arr['X'] = 10;
            arr['L'] = 50;
            arr['C'] = 100;
            arr['D'] = 500;
            arr['M'] = 1000;

            int result = 0;
            int prev = 0;

            for (int i = s.length() - 1; i >= 0; i--) {
                int current = arr[s.charAt(i)];
                result += prev > current ? -current : current;
                prev = current;
            }

            return result;
        }
    }
    
    
# 14 Longest_Common_Prefix

    class Solution {
        public String longestCommonPrefix(String[] strs) {
            String result ="";
            String temp = "";
            int c = 0; //move first point
            boolean check = true;
            while(true){
                for(int i = 0; i<strs.length; i++){ //move second point
                    if(c>=strs[i].length()){
                        check = false;
                        break;
                    }
                    if(i==0){ //temp -> check same Character
                        temp = Character.toString(strs[0].charAt(c));
                    }
                    if(!temp.equals(Character.toString(strs[i].charAt(c)))){
                        check = false;
                        break;
                    }
                    if(i==strs.length-1){
                        result += temp;
                    }
                }
                if(!check){
                    break;
                }
                c++;
            }
            return result;

        }
    }


# 15- 3Sum

    class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        //create result list to store i,j,k
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        
        //sorting nums
        Arrays.sort(nums);
        
        for (int i = 0; i < nums.length - 2; i++) {

            int left = i + 1;
            int right = nums.length - 1;

            if (i > 0 && nums[i] == nums[i-1]) {
                continue; //if nums have same numbers, just check one time.
            } 
            
            while (left < right) {
                int sum = nums[left] + nums[right] + nums[i];
                
                if (sum == 0) {
                    //if sum == 0, store i,j,k
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    left++; //check anoter case
                    right--;
                    //if next number == now number
                    while (nums[left] == nums[left - 1] && left < right) {
                        left++;
                    }
                    while (nums[right] == nums[right + 1] && left < right) {
                        right--;
                    } 
                } else if (sum > 0) {
                    //if sum > 0, right--;
                    right--;
                } else {
                    //if sum < 0, left++;
                    left++;
                }
            }
        }
        
        return result; //return result list

# 19- Remove_Nth_Node_From_End_of_List
        class Solution {
        public ListNode removeNthFromEnd(ListNode head, int n) {
            ListNode slow, fast, curr;
            slow = head; fast = head;
            for (int i = 0; i < n; i++)
                fast = fast.next;
            // n == len
            if (fast == null) {
                head = head.next;
                return head;
            }
            // Move both pointers, until reach tail
            while (fast.next != null) {
                fast = fast.next;
                slow = slow.next;
            }
            curr = slow.next;
            slow.next = curr.next;
            return head;
        }
    }
