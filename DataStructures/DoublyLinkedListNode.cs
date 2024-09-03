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
        public DoublyLinkedListNode<T1, T2> Prev = null;
        public DoublyLinkedListNode<T1,T2> Next = null;
        


        public DoublyLinkedListNode(T1 key, T2 value){
            Key = key;
            Value = value;
        }

        public void RemoveBindings()
        {
            if(Prev!=null)
                Prev.Next = Next;
            if(Next != null)
                Next.Prev = Prev;
            
            Prev=null;
            Next=null;
        }
    }
}