
import math
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        # print(f'type of _children : {type(_children)}')
        self._prev= set(_children)
        self._op = _op
        self.label= label
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        """
            Assumng other either of Value type or number and convert it to Value post typecheck
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data  + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0* out.grad
            other.grad += 1.0* out.grad
        out._backward = _backward
    
        return out
    
    def __neg__(self):
        return self *(-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data *  out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int and flost are supported"
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * self.data **(other -1) * out.grad
        out._backward =_backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self* other**-1
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out= Value(t,(self,), 'tanx')
        
        def _backward():
            self.grad += (1- t**2 )* out.grad
        out._backward = _backward 
        return out
    
    def exp(self):
        x = self.data
        t = math.exp(x)
        out= Value(t,(self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward 
        return out

    def backward(self):
        topological_sorted = []
        visited = set()
        def sort_topological_order(v):
            if v not in visited:
                visited.add(v)
                for child  in v._prev:
                    sort_topological_order(child)
                topological_sorted.append(v)
        sort_topological_order(self)
        self.grad = 1.0
        for node in reversed(topological_sorted):
            node._backward()