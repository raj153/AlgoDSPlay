using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class ListNode
    {
        public string Name;
        public int Val;
        public List<ListNode> Children = new List<ListNode>();
        internal ListNode Next;
        internal ListNode Prev;
        private ListNode curr;

        public ListNode()
        {
        }

        public ListNode(string name)
        {
            this.Name = name;
        }

        public ListNode(int _val)
        {
            Val = _val;
        }

        public ListNode(int _val, List<ListNode> _children)
        {
            Val = _val;
            Children = _children;
        }

        public ListNode(int _val, ListNode curr) : this(_val)
        {
            this.curr = curr;
        }

        public ListNode AddChild(string name)
        {
            ListNode child = new ListNode(name);
            Children.Add(child);
            return this;
        }
        public ListNode(int val, ListNode next, ListNode prev)
        {
            this.Val = val;
            this.Next = next;
            this.Prev = prev;
        }
    }
}
