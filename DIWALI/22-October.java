

## 12. Integer to Roman

    class Solution {
        private static final StringBuilder sb = new StringBuilder(15);

        private static final String[] thou = {"", "M", "MM", "MMM"};
        private static final String[] hund = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC","DCCC","CM"};
        private static final String[] tens = {"", "X", "XX", "XXX", "XL", "L", "LX","LXX","LXXX","XC"};
        private static final String[] ones = {"", "I", "II", "III", "IV", "V", "VI","VII","VIII","IX"};

        public String intToRoman(int num) {
            sb.setLength(0);

            sb.append(thou[num/1000]);
            sb.append(hund[(num%1000)/100]);
            sb.append(tens[(num%100)/10]);
            sb.append(ones[num%10]);

            return sb.toString();
        }
    }


## 237. Delete Node in a Linked List

    class Solution {
        public void deleteNode(ListNode node) {
            node.val=node.next.val;
            node.next=node.next.next;
        }
    }


## 2095. Delete the Middle Node of a Linked List

    class Solution {
        public ListNode deleteMiddle(ListNode head) {

            //Code here....
            if(head == null || head.next == null)
                return null;    //Checking corner case

            ListNode curr = head;
            curr = middle(curr);    //Finding the element previous to the middle node
            curr.next = curr.next.next; //Removing the middle node

            return head;

        }

        public ListNode middle(ListNode head)
        {
            ListNode slow = head, fast = head,prev = null;
            while(fast != null && fast.next != null)
            {
                prev = slow;
                slow = slow.next;
                fast = fast.next.next;
            }

            return prev;
        }
    }


## 1531. String Compression II

    class Solution {
        private int[][] dp;
        private char[] chars;
        private int n;

        public int getLengthOfOptimalCompression(String s, int k) {
            this.chars = s.toCharArray();
            this.n = s.length();
            this.dp = new int[n][k+1];
            for (int[] row: dp) {
                Arrays.fill(row, -1);
            }
            return dp(0, k);
        }

        private int dp(int i, int k) {
            if (k < 0) return n;
            if (n <= i + k) return 0;

            int ans = dp[i][k];
            if (ans != -1) return ans; 
            ans = dp(i + 1, k - 1);
            int length = 0, same = 0, diff = 0;

            for (int j=i; j < n && diff <= k; j++) {

                if (chars[j] == chars[i]) {
                    same++;
                    if (same <= 2 || same == 10 || same == 100) length++;
                } else {
                    diff++; 
                }
                ans = Math.min(ans, length + dp(j + 1, k - diff)); 
            }
            dp[i][k] = ans;
            return ans;
        }
    }


## 1335. Minimum Difficulty of a Job Schedule

    class Solution {
        public int minDifficulty(int[] jobDifficulty, int d) {
            int len = jobDifficulty.length;
            if (d > len) return -1;
            int[][] minDifficulty = new int[d][len];
            for (int i = 1; i < d; i++) {
                Arrays.fill(minDifficulty[i], Integer.MAX_VALUE);
            }
            int maxDifficulty = 0;
            for (int i = 0; i <= len - d; i++) {
                maxDifficulty = Math.max(maxDifficulty, jobDifficulty[i]);
                minDifficulty[0][i] = maxDifficulty;
            }
            for (int day = 1; day < d; day++) {
                for (int to = day; to <= len - d + day; to++) {
                    int currentDayDifficulty = jobDifficulty[to];
                    int result = Integer.MAX_VALUE;
                    for (int j = to - 1; j >= day - 1; j--) {
                        result = Math.min(result, minDifficulty[day - 1][j] + currentDayDifficulty);
                        currentDayDifficulty = Math.max(currentDayDifficulty, jobDifficulty[j]);
                    }
                    minDifficulty[day][to] = result;
                }   
            }
            return minDifficulty[d - 1][len - 1];
        }
    }



## 1832. Check if the Sentence Is Pangram
    class Solution {
        public boolean checkIfPangram(String sentence) {
            if(sentence.length()<26) return false;
             HashSet<Character> set = new HashSet<>();
            for(char c:sentence.toCharArray()){
                set.add(c);
            }

            return set.size()==26;

        }
    }


## 38. Count and Say

    class Solution {
        public String countAndSay(int n) {
            String s = "1";
            for(int i=2;i<=n;i++){
                s = countIndex(s);
            }
            return s;
        }

        public String countIndex(String s){
            StringBuilder sb = new StringBuilder();
            char c = s.charAt(0);
            int count = 1;
            for(int i=1;i<s.length();i++){
                if(s.charAt(i) == c)
                    count++;
                else{
                    sb.append(count);
                    sb.append(c);
                    c = s.charAt(i);
                    count=1;
                }
            }
            sb.append(count);
            sb.append(c);
            return sb.toString();
        }
    }



## 692. Top K Frequent Words

    class Solution {
        public List<String> topKFrequent(String[] words, int k) {
                TreeMap<String, Integer> map = new TreeMap<>(String::compareTo);
                Arrays.stream(words).forEach(x -> map.put(x, map.getOrDefault(x, 0) + 1));
                return map.entrySet().stream()
                    .sorted((o1, o2) -> Integer.compare(o2.getValue(), o1.getValue()))
                    .map(Map.Entry::getKey)
                    .limit(k)
                    .collect(Collectors.toList());
            }
    }



## 12. Integer to Roman

    class Solution {
        private static final StringBuilder sb = new StringBuilder(15);

        private static final String[] thou = {"", "M", "MM", "MMM"};
        private static final String[] hund = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC","DCCC","CM"};
        private static final String[] tens = {"", "X", "XX", "XXX", "XL", "L", "LX","LXX","LXXX","XC"};
        private static final String[] ones = {"", "I", "II", "III", "IV", "V", "VI","VII","VIII","IX"};

        public String intToRoman(int num) {
            sb.setLength(0);

            sb.append(thou[num/1000]);
            sb.append(hund[(num%1000)/100]);
            sb.append(tens[(num%100)/10]);
            sb.append(ones[num%10]);

            return sb.toString();
        }
    }  





## 2289. Steps to Make Array Non-decreasing

    class Solution {
        public int totalSteps(int[] nums) {
            int max=0;
            Stack<Integer> st = new Stack<>();
            int[] dp = new int[nums.length];
            st.push(nums.length-1);
            for(int i=nums.length-2;i>=0;i--){
                if(nums[st.peek()]>=nums[i]){
                    st.push(i);
                }else{
                    while(!st.isEmpty() && nums[st.peek()]<nums[i]){
                        dp[i] = Math.max(dp[i]+1,dp[st.peek()]);
                        st.pop();
                    }
                    max = Math.max(max,dp[i]);
                    st.push(i);
                }
            }
            return max;
        }
    }
