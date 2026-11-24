"""借助字节码反推闭包 / 嵌套函数引用的全局名。"""

import dis
import ast
import types
from collections import defaultdict

from codeharvest.analysis.ast_tools import parse_ast

def instruction_matches_node(instruction: dis.Instruction, node: ast.AST) -> bool:
    return (
        instruction.positions is not None
        and instruction.positions.lineno == node.lineno  
        and instruction.positions.col_offset == node.col_offset  
        and instruction.positions.end_lineno == node.end_lineno  
        and instruction.positions.end_col_offset == node.end_col_offset  
    )

def map_names_to_nodes(module_ast: ast.Module) -> dict[str, list[ast.Name]]:
    id_to_nodes = defaultdict(list)
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Name):
            id_to_nodes[node.id].append(node)
    return id_to_nodes

def code_object_arg_names(module_ast: ast.Module, code_obj: types.CodeType) -> list[str]:
    argument_names = []
    code_obj_name = code_obj.co_name
    code_obj_firstlineno = code_obj.co_firstlineno
    for node in ast.walk(module_ast):
        if (
            isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)
        ) and node.name == code_obj_name:
            def_lineno_delta = [
                idx
                for idx, line in enumerate(ast.unparse(node).split("\n"))
                if line.strip().startswith("def")
                or line.strip().startswith("async def")
            ]
            assert len(def_lineno_delta) >= 1, ast.unparse(node)
            def_lineno_delta = def_lineno_delta[0]
            def_lineno = node.lineno - def_lineno_delta
            if def_lineno == code_obj_firstlineno:
                argument_names = (
                    [x.arg for x in node.args.args]
                    + [x.arg for x in node.args.kwonlyargs]
                    + [x.arg for x in [node.args.vararg, node.args.kwarg] if x]
                )
        if isinstance(node, ast.Lambda):
            argument_names = [x.arg for x in node.args.args]
    return argument_names

def names_from_const_code(module_ast: ast.Module, code_obj: types.CodeType) -> list[str]:
    argument_names = code_object_arg_names(module_ast, code_obj)

    fast_stores: list[str] = []

    code_bytecode = dis.Bytecode(code_obj)

    id_to_nodes: dict[str, list[ast.Name]] = map_names_to_nodes(module_ast)

    global_access_symbols = []
    for instruction in code_bytecode:
        if instruction.opname == "LOAD_GLOBAL":
            instruct_arg_name: str = instruction.argval  
            potential_nodes = id_to_nodes.get(instruct_arg_name, [])
            for potential_node in potential_nodes:
                if instruction_matches_node(instruction, potential_node):
                    
                    global_access_symbols.append(instruct_arg_name)

        elif instruction.opname == "STORE_FAST":
            instruct_arg_name: str = instruction.argval  
            fast_stores.append(instruct_arg_name)

        elif instruction.opname == "LOAD_FAST":
            instruct_arg_name: str = instruction.argval  
            potential_nodes = id_to_nodes.get(instruct_arg_name, [])
            for potential_node in potential_nodes:
                if instruction_matches_node(instruction, potential_node):
                    if instruct_arg_name not in fast_stores:
                        
                        
                        
                        
                        global_access_symbols.append(instruct_arg_name)

        elif instruction.opname == "LOAD_NAME":
            instruct_arg_name: str = instruction.argval  
            potential_nodes = id_to_nodes.get(instruct_arg_name, [])
            for potential_node in potential_nodes:
                if instruction_matches_node(instruction, potential_node):
                    
                    global_access_symbols.append(instruct_arg_name)

        elif instruction.opname == "LOAD_CONST":
            if isinstance(instruction.argval, types.CodeType):
                global_access_symbols.extend(
                    names_from_const_code(module_ast, instruction.argval)
                )

        global_access_symbols = [
            g for g in global_access_symbols if g not in argument_names
        ]
    return global_access_symbols

def definition_global_names(
    func_class_ast: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> list[str]:
    
    
    func_class_ast_str = ast.unparse(func_class_ast)
    module_ast = parse_ast(func_class_ast_str, add_parents=False)
    module_ast = parse_ast(ast.unparse(module_ast), add_parents=False)

    try:
        module_ast_compiled = compile(ast.unparse(module_ast), "<string>", "exec")
    except SyntaxError:
        print(f"Syntax error in {ast.unparse(module_ast)}")
        return []

    module_bytecode = dis.Bytecode(module_ast_compiled)

    global_access_symbols = []
    for instruction in module_bytecode:
        if instruction.opname == "LOAD_CONST":
            if isinstance(instruction.argval, types.CodeType):
                code = instruction.argval
                global_access_symbols.extend(names_from_const_code(module_ast, code))

    function_name = func_class_ast.name
    global_access_symbols = [
        symbol for symbol in global_access_symbols if symbol != function_name
    ]
    return global_access_symbols
