

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





86. Partition List

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
