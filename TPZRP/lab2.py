

class ChordNode:
    successor = 0
    predecessor = 0
    key = 0

    def __init__(self, name, next, prev):
        if name != 0:
            self.name = name  # имя узла
            self.successor = next  # следующий узел
            self.predecessor = prev  # следующий узел
            self.key += 1  # следующий узел
        else:
            self.name = name  # имя узла
            self.successor = None  # следующий узел
            self.predecessor = None  # следующий узел
            self.key += 1  # следующий узел

    def display_info(self):
        print(f"Name: {self.name}  Next: {self.successor}")

    @staticmethod
    def print(node):
        print(f"[{node.name}]->")


node1 = ChordNode(1, 'Null', 'Null')
node1.display_info()
node2 = ChordNode(2, node1, node1)
node2.display_info()
node3 = ChordNode(3, node1, node2)
node3.display_info()

ChordNode.print(node1)
