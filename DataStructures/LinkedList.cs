using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class LinkedList
    {
        public int Value;

        public LinkedList Next;

        public LinkedList(int value){
            this.Value = value;
            this.Next=null;
        }
    }
}