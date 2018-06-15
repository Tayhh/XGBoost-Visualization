import sys
import re
import math
from graphviz import Digraph
import matplotlib.pyplot as plt
import xgboost as xgb
plt.switch_backend('agg')




def Amt(depth_index):
    """
    count the number of  lines, belonging to the tree,which needs to be showed

    :param depth_index:the depth of the tree
    :return:amt:the number of  lines belonging to the tree
    """
    depthIndex = int(depth_index)
    amt = 2
    for i in range(depthIndex):
        amt += 2 ** (i + 1)
    return amt


def treePosition(tree_index, depth_index):
    """
    locate the start position ,end position of the tree, which needs to be showed

    :param tree_index: the index of the tree
    :param depth_index:the depth of the tree
    :return:tree_position:list,the fist element is the start position,the second element is the end position
    """
    treeIndex = int(tree_index)
    amt = Amt(depth_index)
    tree_position = []
    startPosition = treeIndex * amt
    tree_position.append(startPosition)
    endPosition = treeIndex * amt + 64
    tree_position.append(endPosition)
    return tree_position


def getNodeEdgeInfo(model_path_from, model_path_to, feature_map, tree_index, depth_index):
    """
    get the information of the node and that of the edge

    :param model_path_from:the original model file
    :param model_path_to:the path of the model file
    :param feature_map:the map of the feature
    :param tree_index:the index of the tree
    :param depth_index:the depth of the tree
    :return:[nodes_name, nodes, edges_from, edges_toLeft, edges_toRight, leaf_name, leaf]
    nodes_name:the index of the node
    nodes:the context of the node
    edges_from:the node index ,which edge comes from
    edgedges_toLeft:the left node index,which the edge arrives to
    edges_toRight:the right node index, which the edge arrives to
    leaf_name: the index of the leaf node
    leaf: the value of the leaf node, transformed by sigmod function

    the nodes is the list of the node ,the edges is the list of the edge
    """

    nodes_name = []
    nodes = []
    leaf_name = []
    leaf = []
    edges_from = []
    edges_toLeft = []
    edges_toRight = []

    model = xgb.Booster(model_file=model_path_from)
    model.dump_model(model_path_to, fmap=feature_map)
    tp = treePosition(tree_index, depth_index)

    with open(model_path_to) as f:
        lines = f.readlines()
        sub_lines = lines[tp[0]:tp[1]]
        for line in sub_lines:
            line_striped = line.strip()
            line_splited = re.split(r":|[|]|\s|=|,", line_striped)
            if (len(line_splited) == 2):
                continue
            if (len(line_splited) == 8):
                line_splited[1] = line_splited[1].strip("[").strip("]")
                nodes_name.append(line_splited[0])
                nodes.append(line_splited[1])
                edges_from.append(line_splited[0])
                # TODO repeat edges_from when create the graph by dot, it's element is repeated ,for example, before:[1,2,3], after:[1,1,2,2,3,3]
                edges_toLeft.append(line_splited[3])
                edges_toRight.append(line_splited[5])
                continue
            if (len(line_splited) == 3):
                leaf_name.append(line_splited[0])
                x = float(line_splited[2])
                leaf.append(str(round(100.0 / (1 + math.e ** -x), 2)) + "%")
                continue
    return nodes_name, nodes, leaf_name, leaf, edges_from, edges_toLeft, edges_toRight


def CreateGraph(model_path_from, model_path_to, feature_map, tree_index, depth_index, picture_path):
    """

    :param model_path_to: the path of the model file
    :param tree_index: the index of the tree
    :param depth_index: the depth of the tree
    :param picture_path: the path of the picture
    :return:save the picture of the tree needing to be showed
    """
    NodesName, Nodes, LeafName, Leaf, EdgesFrom, EdgesToLeft, EdgesToRight = getNodeEdgeInfo(model_path_from, model_path_to, feature_map, tree_index, depth_index)
    dot = Digraph()
    for i in range(len(NodesName)):
        dot.node(NodesName[i], Nodes[i])
    for i in range(len(LeafName)):
        dot.node(LeafName[i], Leaf[i])
    for i in range(len(EdgesFrom)):
        dot.edge(EdgesFrom[i], EdgesToLeft[i], label="Yes")
    for i in range(len(EdgesFrom)):
        dot.edge(EdgesFrom[i], EdgesToRight[i], label="No")
    if picture_path != None:
        dot.save(filename="tree" + ".gv", directory=picture_path)
        dot.render(filename="tree" + ".gv", directory=picture_path, view=True)


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("argument error!")
        sys.exit(0)
    modelPathFrom = sys.argv[1]
    modelPathTo = sys.argv[2]
    featureMap = sys.argv[3]
    treeIndex = sys.argv[4]
    depthIndex = sys.argv[5]
    picturePath = sys.argv[6]
    CreateGraph(modelPathFrom, modelPathTo, featureMap, treeIndex, depthIndex, picturePath)










