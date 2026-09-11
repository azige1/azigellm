"""把 AST 语句反解析回源码，并保留原始注释。"""

import ast

def unparse_stmt_keep_comments(file_content: str, stmt: ast.stmt) -> str:

    file_content_lines = file_content.split("\n")
    stmt_start_lineno = stmt.lineno - 1
    stmt_start_col_offset = stmt.col_offset
    
    stmt_end_lineno = stmt.end_lineno - 1  
    stmt_end_col_offset = stmt.end_col_offset  

    stmt_lines = file_content_lines[stmt_start_lineno : stmt_end_lineno + 1]

    stmt_lines[0] = stmt_lines[0][stmt_start_col_offset:]
    stmt_lines[-1] = stmt_lines[-1][:stmt_end_col_offset]

    return "\n".join(stmt_lines)
