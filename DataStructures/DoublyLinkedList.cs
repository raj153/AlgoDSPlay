using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class DoublyLinkedList<T1, T2>
    {
        public DoublyLinkedListNode<T1,T2> Head = null;
        public DoublyLinkedListNode<T1,T2> Tail = null;

        public void SetHeadTo(DoublyLinkedListNode<T1, T2> node){
            if( Head == node)
                return;
            else if(Head == null){
                Head = node;
                Tail=node;
            }else if(Head == Tail){
                Tail.prev=node;
                Head = node;
                Head.next = Tail;
            }else{
                if(Tail == node ){
                    RemoveTail();
                }
                node.RemoveBindings();
                Head.prev=node;
                node.next = Head;
                Head = node;
            }
                
        }
        

        public void RemoveTail()
        {
            if(Tail == null)
                return;
            if( Tail == Head)
            {
                Head = null;
                Tail = null;
                return;
                
            }
            Tail = Tail.prev;
            Tail.next= null;
        }
    }
}