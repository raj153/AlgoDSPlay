using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class DoublyLinkedListNode<T1, T2>
    {
        public T1 Key;
        public T2 Value;
        public DoublyLinkedListNode<T1, T2> prev = null;
        public DoublyLinkedListNode<T1,T2> next = null;


        public DoublyLinkedListNode(T1 key, T2 value){
            Key = key;
            Value = value;
        }

        public void RemoveBindings()
        {
            if(prev!=null)
                prev.next = next;
            if(next != null)
                next.prev = prev;
            
            prev=null;
            next=null;
        }
    }
}