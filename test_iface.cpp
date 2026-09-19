interface S { int x; };
S s1;
const int& get() { return s1.x; }
int main() { return (int)get(); }
