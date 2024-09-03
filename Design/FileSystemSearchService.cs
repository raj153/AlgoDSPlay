using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    //Using Specification pattern to suppoer And/Or/Lessthan/Greaterthan 
    public class FileSystemSearchService
    {
        public class File
        {
            public string Name
            {
                get;
                set;
            }
            public bool IsFile
            {
                get;
                set;
            }
            public string Content
            {
                get;
                set;
            }
            public int Size
            {
                get;
                set;
            }
            public Dictionary<string, File> Childeren
            {
                get;
                set;
            }
            public File(string name, int size, bool isfile = true)
            {
                this.Name = name;
                this.Size = size;
                this.IsFile = isfile;
                Childeren = new Dictionary<string, File>();
            }
        }
        public enum Operator
        {
            equal,
            greaterThan,
            greaterThanEqual,
            lessThan,
            lessThanEqual
        }
        public interface ISpecification
        {
            bool IsSatisfiedBy(File f);
            ISpecification And(ISpecification other);
        }
        public abstract class CompositeSpecification : ISpecification
        {
            public abstract bool IsSatisfiedBy(File file);
            public ISpecification And(ISpecification other)
            {
                return new AndSpecification(this, other);
            }
        }
        public class AndSpecification : CompositeSpecification
        {
            private ISpecification leftCondition;
            private ISpecification rightCondition;

            public AndSpecification(ISpecification left, ISpecification right)
            {
                leftCondition = left;
                rightCondition = right;
            }
            public override bool IsSatisfiedBy(File candidate)
            {
                return leftCondition.IsSatisfiedBy(candidate) && rightCondition.IsSatisfiedBy(candidate);
            }

        }
        public class NameSearchCriteria : CompositeSpecification
        {
            private string Name
            {
                get;
                set;
            }
            public NameSearchCriteria(string name)
            {
                this.Name = name;
            }
            public override bool IsSatisfiedBy(File f)
            {
                return f.Name.Equals(this.Name);
            }
        }

        public class SizeSearchCriteria : CompositeSpecification
        {
            private int Size
            {
                get;
                set;
            }
            private Operator SizeOperator
            {
                get;
                set;
            }
            public SizeSearchCriteria(int size, Operator op)
            {
                this.Size = size;
                this.SizeOperator = op;
            }
            public override bool IsSatisfiedBy(File f)
            {
                if (this.SizeOperator == Operator.equal)
                {
                    return f.Size.CompareTo(f.Size) == 0;
                }
                return false;
            }
        }
        public class FileSearchService
        {
            private List<File> Files
            {
                get;
                set;
            }
            public FileSearchService(List<File> files)
            {
                this.Files = files;
            }
            public List<File> Search(ISpecification search)
            {
                List<File> res = new List<File>();
                foreach (File f in this.Files)
                {
                    if (search.IsSatisfiedBy(f))
                    {
                        res.Add(f);
                    }
                }
                return res;
            }
        }
        public class FileSystem
        {
            public static void Main(string[] args)
            {
                File rootDir = new File("home", 10, false);

                File f1 = new File("abc.txt", 2, false);
                File f2 = new File("bcd.txt", 5, false);
                File f3 = new File("def.png", 15, false);
                File f4 = new File("gef.jpeg", 20, false);

                rootDir.Childeren.Add(f1.Name, f1);
                rootDir.Childeren.Add(f2.Name, f2);
                rootDir.Childeren.Add(f3.Name, f3);
                rootDir.Childeren.Add(f4.Name, f4);

                NameSearchCriteria name = new NameSearchCriteria("abc.txt");
                SizeSearchCriteria size = new SizeSearchCriteria(10, Operator.equal);
                ISpecification search = name.And(size);

                List<File> files = new List<File>();
                files.AddRange(rootDir.Childeren.Values);
                FileSearchService fs = new FileSearchService(files);
                fs.Search(search);
            }
        }

    }
}