def is_point_inside_rectangle(x, y, rect_x1, rect_y1, rect_x2, rect_y2):
    """
    長方形の範囲内に点があるかどうかを判定する関数

    Parameters:
    - x, y: チェックする点の座標
    - rect_x1, rect_y1: 長方形の1つの角の座標
    - rect_x2, rect_y2: 長方形の対角の角の座標

    Returns:
    - 真(True)：点が長方形の範囲内にある
    - 偽(False)：点が長方形の範囲外にある
    """
    return rect_x1 <= x <= rect_x2 and rect_y1 <= y <= rect_y2

# 例: 長方形の範囲 (0, 0) から (5, 5) に点 (3, 3) が含まれているかを判定

x=[0,1,2,3,4,5,6,7]
y=[0,1,2,3,4,5,6,7]
result = is_point_inside_rectangle(x, y, 0, 0, 5, 5)

if result:
    print("点は長方形の範囲内にあります。")
else:
    print("点は長方形の範囲外にあります。")