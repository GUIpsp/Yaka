package pt.ul.fc.bytec;

import com.google.common.base.Strings;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Label;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.*;
import org.objectweb.asm.tree.analysis.Analyzer;
import org.objectweb.asm.tree.analysis.AnalyzerException;
import org.objectweb.asm.tree.analysis.BasicInterpreter;
import org.objectweb.asm.tree.analysis.BasicValue;
import org.objectweb.asm.util.Textifier;
import org.objectweb.asm.util.TraceMethodVisitor;
import pt.ul.fc.bytec.MethodDescriptor.ArrayDescriptor;
import pt.ul.fc.bytec.tests.Tests;

import java.io.*;
import java.util.*;

public class ByteC {

    private final ConversionConfiguration conf;

    public ByteC(ConversionConfiguration conf) {
        this.conf = conf;
    }

    public static void main(String[] args) throws IOException {
        Class<Tests> clazz = Tests.class;
        new ByteC(new ConversionConfiguration() {
            @Override
            public String getModifiers(ClassNode clazz, MethodNode method) {
                return "";
            }
        }).convertClass(clazz);
    }

    public File convertClass(Class<?> clazz) throws IOException {
        String canonicalName = clazz.getCanonicalName();
        return convertClass(canonicalName);
    }

    public File convertClass(String canonicalName) throws IOException {
        File ret = new File(fixPackagePath(canonicalName) + conf.getExtension());
        Deque<String> open = new ArrayDeque<>();
        open.add(canonicalName);
        HashSet<String> closed = new HashSet<>();
        while (!open.isEmpty()) {
            String curr = open.pop();
            if (closed.contains(curr)) continue;
            closed.add(curr);
            Set<String> set = doClass(curr);
            open.addAll(set);
        }
        return ret;
    }

    private Set<String> doClass(String canonicalName) throws IOException {
        File file = new File(fixPackagePath(canonicalName) + conf.getExtension());
        ClassNode classNode = new ClassNode();
        ClassReader cr = new ClassReader(canonicalName);
        cr.accept(classNode, 0);
        String name = classNode.name;
        String className = fixPackagePath(name);
        File header = new File(className + ".h");
        FileOutputStream out = new FileOutputStream(file);
        FileOutputStream headerOut = new FileOutputStream(header);

        OutputStreamWriter headerWriter = new OutputStreamWriter(headerOut);

        PrintWriter headerPrinter = new PrintWriter(headerWriter);

        String header_guard = mangle(canonicalName, "HEADER_GUARD");
        String impl_guard = mangle(canonicalName, "IMPL_GUARD");

        headerPrinter.printf("#ifndef %s\n#define %s\n", header_guard, header_guard);


        PrintStream p = new PrintStream(out);
        p.printf("#ifndef %s\n#define %s\n", impl_guard, impl_guard);
        p.println("#include <stdint.h>");
        p.println("#include <stdbool.h>");
        p.println();
        p.println(conf.getExtraHeaders());

        Set<String> reqs = new HashSet<>();

        StringWriter body = new StringWriter();
        PrintWriter bodyWriter = new PrintWriter(body);


        for (MethodNode method : classNode.methods) {
            try {
                reqs.addAll(convert(classNode, method, bodyWriter, headerPrinter));
            } catch (Exception e) {
                new Exception("could not conv " + method.name, e).printStackTrace();
            }
        }
        bodyWriter.close();
        body.close();

        for (String req : reqs) {
            p.println("#include \"" + req + ".h\"");
        }

        for (String req : reqs) {
            p.println("#include \"" + req + conf.getExtension() + "\"");
        }

        p.println();
        for (FieldNode field : classNode.fields) {
            Object o = MethodDescriptor.decodeDescriptor(field.desc, new int[]{0});
            String prefix = conf.getFieldModifiers(classNode, field);
            prefix = Strings.nullToEmpty(prefix);
            if (!prefix.isEmpty()) {
                prefix += " ";
            }
            p.printf("%s%s %s;\n", prefix, MethodDescriptor.stringify(o), mangle(classNode.name, field.name));
        }
        p.println();

        p.print(body);
        headerPrinter.println("#endif");
        p.print("#endif");
        p.close();
        out.close();


        headerPrinter.close();
        headerWriter.close();
        headerOut.close();

        return reqs;
    }

    public static String fixPackagePath(String name) {
        return name.replace('/', '.');
    }

    public static String mangle(String owner, String name) {
        return owner.replaceAll("[./]", "_DOT_") + "_AT_" + name;
    }

    public Set<String> convert(ClassNode owner, MethodNode method, PrintWriter pFinal, PrintWriter headerWriter) {
        Set<String> requiredFiles = new HashSet<>();
        if (method.name.startsWith("<")) {
            return requiredFiles;
        }
        HashSet<String> gennedVars = new HashSet<>();
//        System.out.println(method.name);
//        System.out.println(method.desc);
        {
            // Print method prefixes
            String prefix = conf.getModifiers(owner, method);
            prefix = Strings.nullToEmpty(prefix);
            if (!prefix.isEmpty()) {
                pFinal.print(prefix);
                pFinal.print(" ");
            }
        }
        MethodDescriptor methodDescriptor = printSignature(owner.name, method, pFinal, gennedVars);

        printSignature(owner.name, method, headerWriter, gennedVars);
        headerWriter.println(";\n");

        pFinal.println("{");

        {
            if (method.invisibleAnnotations != null)
                for (AnnotationNode invisibleAnnotation : method.invisibleAnnotations) {
                    if ("Lpt/ul/fc/bytec/CImplementation;".equals(invisibleAnnotation.desc)) { //TODO: this is very hacky
                        pFinal.println(invisibleAnnotation.values.get(1));
                        pFinal.println("}");
                        return requiredFiles;
                    }
                }
        }

        {
            for (Object type : MethodDescriptor.PRIMITIVES) {
                String delimiter = ", ";
                if (type instanceof ArrayDescriptor) {
                    delimiter += "*";
                }
                StringJoiner j = new StringJoiner(delimiter, MethodDescriptor.stringify(type) + " ", ";\n");
                j.setEmptyValue("");
                for (int i = 0; i < method.maxLocals; i++) {
                    String genned = "a" + MethodDescriptor.getSafeName(type) + i;
                    if (!gennedVars.contains(genned)) {
                        gennedVars.add(genned);
                        j.add(genned);
                    }
                }
                pFinal.print(j);
            }
        }

        {
            for (Object type : MethodDescriptor.PRIMITIVES) {
                String delimiter = ", ";
                if (type instanceof ArrayDescriptor) {
                    delimiter += "*";
                }
                StringJoiner j = new StringJoiner(delimiter, MethodDescriptor.stringify(type) + " ", ";\n");
                j.setEmptyValue("");
                for (int i = 0; i < method.maxStack; i++) {
                    String genned = "s" + MethodDescriptor.getSafeName(type) + i;
                    if (!gennedVars.contains(genned)) {
                        gennedVars.add(genned);
                        j.add(genned);
                    }
                }
                pFinal.print(j);
            }
        }

        InsnList insnList = method.instructions;

        // Pre-initialize label names

        Map<Label, String> labelIds = new HashMap<>();

        {
            int curr = 0;
            for (int i = 0; i < insnList.size(); i++) {
                AbstractInsnNode label = insnList.get(i);
                if (label instanceof LabelNode) {
                    labelIds.put(((LabelNode) label).getLabel(), "L" + curr++);
                }
            }
        }


        //Obtain the control flow edges
        CFGAnalyser analyzer = new CFGAnalyser();
        try {
            analyzer.analyze(owner.name, method);
        } catch (AnalyzerException ignored) {

        }

        Multimap<Integer, Integer> cfg = analyzer.edges;

        CharSequence[] translated = new CharSequence[insnList.size()];

        Object[] starterLocalTypes = new Object[method.maxLocals];

        for (int i = 0; i < methodDescriptor.args.length; i++) {
            starterLocalTypes[i] = methodDescriptor.args[i];
        }

        // Explore the graph and convert all instructions
        ArrayDeque<OpenEdge> open = new ArrayDeque<>();
        HashSet<Integer> closed = new HashSet<>();
        open.add(new OpenEdge(new ArrayDeque<>(), starterLocalTypes, 0));
        try {
            while (!open.isEmpty()) {
                OpenEdge openEdge = open.removeFirst();
                int curr = openEdge.index;
                if (closed.contains(curr)) {
                    continue;
                }
                closed.add(curr);
                AbstractInsnNode instruction = insnList.get(curr);
                // Represents the stack at a given point of execution
                ArrayDeque<Object> stack = new ArrayDeque<>(openEdge.stack);
                int stackHeight = stack.size();
                // Represents the types of the local variables
                Object[] localTypes = openEdge.localTypes.clone();

                StringWriter stringWriter = new StringWriter();
                PrintWriter p = new PrintWriter(stringWriter);
                int type = instruction.getType();
                int opcode = instruction.getOpcode();
                switch (type) {
                    case AbstractInsnNode.INSN: {
                        switch (opcode) {
                            case Opcodes.IMUL: {
                                p.printf("sI%d = sI%d * sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.IDIV: {
                                p.printf("sI%d = sI%d / sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.IREM: {
                                p.printf("sI%d = sI%d %% sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ISUB: {
                                p.printf("sI%d = sI%d - sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.IADD: {
                                p.printf("sI%d = sI%d + sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ISHL: {
                                p.printf("sI%d = ((uint32_t)sI%d) << sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ISHR: {
                                p.printf("sI%d = sI%d >> sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.IUSHR: {
                                p.printf("sI%d = ((uint32_t)sI%d) >> sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.IOR: {
                                p.printf("sI%d = sI%d | sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.IAND: {
                                p.printf("sI%d = sI%d & sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.IXOR: {
                                p.printf("sI%d = sI%d ^ sI%d;\n", stackHeight - 2, stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ICONST_M1: {
                                p.printf("sI%d = %d;\n", stackHeight, -1);
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ICONST_0: {
                                p.printf("sI%d = %d;\n", stackHeight, 0);
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ICONST_1: {
                                p.printf("sI%d = %d;\n", stackHeight, 1);
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ICONST_2: {
                                p.printf("sI%d = %d;\n", stackHeight, 2);
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ICONST_3: {
                                p.printf("sI%d = %d;\n", stackHeight, 3);
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ICONST_4: {
                                p.printf("sI%d = %d;\n", stackHeight, 4);
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ICONST_5: {
                                p.printf("sI%d = %d;\n", stackHeight, 5);
                                stack.push(Integer.class);
                                break;
                            }
                            // Returns
                            case Opcodes.IRETURN: {
                                p.printf("return sI%d;\n", stackHeight - 1);
                                break;
                            }
                            case Opcodes.RETURN: {
                                p.printf("return;\n");
                                break;
                            }
                            // Array loads and stores
                            case Opcodes.IALOAD: {
                                ArrayDescriptor t = new ArrayDescriptor(Integer.class);
                                p.printf("sI%d = %s[sI%d];\n", stackHeight - 2, cast("sL" + (stackHeight - 2), t), stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.push(t);
                                break;
                            }
                            case Opcodes.IASTORE: {
                                ArrayDescriptor t = new ArrayDescriptor(Integer.class);
                                p.printf("%s[sI%d] = sI%d;\n", cast("sL" + (stackHeight - 3), t), stackHeight - 2, stackHeight - 1);
                                stack.pop();
                                stack.pop();
                                stack.pop();
                                break;
                            }
                            //Dups
                            case Opcodes.DUP: {
                                Object t = stack.peek();
                                emitCopy(p, t, stackHeight - 1, stackHeight);
                                stack.push(t);
                                break;
                            }
                            case Opcodes.DUP_X1: {
                                Object t1 = stack.pop();
                                Object t2 = stack.pop();
                                emitCopy(p, t1, stackHeight - 1, stackHeight);
                                emitCopy(p, t2, stackHeight - 2, stackHeight - 1);
                                emitCopy(p, t1, stackHeight, stackHeight - 2);
                                stack.push(t1);
                                stack.push(t2);
                                stack.push(t1);
                                break;
                            }
                            case Opcodes.DUP_X2: {
                                Object t1 = stack.pop();
                                Object t2 = stack.pop();
                                Object t3 = stack.pop();
                                emitCopy(p, t1, stackHeight - 1, stackHeight);
                                emitCopy(p, t2, stackHeight - 2, stackHeight - 1);
                                emitCopy(p, t3, stackHeight - 3, stackHeight - 2);
                                emitCopy(p, t1, stackHeight, stackHeight - 3);
                                stack.push(t1);
                                stack.push(t3);
                                stack.push(t2);
                                stack.push(t1);
                                break;
                            }
                            case Opcodes.DUP2: {
                                Object t1 = stack.pop();
                                Object t2 = stack.pop();
                                emitCopy(p, t1, stackHeight - 1, stackHeight + 1);
                                emitCopy(p, t1, stackHeight - 2, stackHeight);
                                stack.push(t2);
                                stack.push(t1);
                                stack.push(t2);
                                stack.push(t1);
                                break;
                            }
                            case Opcodes.DUP2_X1: {
                                Object t1 = stack.pop();
                                Object t2 = stack.pop();
                                Object t3 = stack.pop();
                                emitCopy(p, t1, stackHeight - 1, stackHeight + 1);
                                emitCopy(p, t2, stackHeight - 2, stackHeight);
                                emitCopy(p, t3, stackHeight - 3, stackHeight - 1);
                                emitCopy(p, t1, stackHeight + 1, stackHeight - 2);
                                emitCopy(p, t2, stackHeight, stackHeight - 3);
                                stack.push(t2);
                                stack.push(t1);
                                stack.push(t3);
                                stack.push(t2);
                                stack.push(t1);
                                break;
                            }
                            case Opcodes.DUP2_X2: {
                                Object t1 = stack.pop();
                                Object t2 = stack.pop();
                                Object t3 = stack.pop();
                                Object t4 = stack.pop();
                                emitCopy(p, t1, stackHeight - 1, stackHeight + 1);
                                emitCopy(p, t2, stackHeight - 2, stackHeight);
                                emitCopy(p, t3, stackHeight - 3, stackHeight - 1);
                                emitCopy(p, t4, stackHeight - 4, stackHeight - 2);
                                emitCopy(p, t1, stackHeight + 1, stackHeight - 3);
                                emitCopy(p, t2, stackHeight, stackHeight - 4);
                                stack.push(t2);
                                stack.push(t1);
                                stack.push(t4);
                                stack.push(t3);
                                stack.push(t2);
                                stack.push(t1);
                                break;
                            }
                            default: {
                                p.println("//Unknown instruction, ignored: ");
                            }
                        }
                        break;
                    }
                    case AbstractInsnNode.INT_INSN: {
                        IntInsnNode insn = (IntInsnNode) instruction;
                        switch (opcode) {
                            case Opcodes.BIPUSH:
                            case Opcodes.SIPUSH: {
                                p.printf("sI%d = %d;\n", stackHeight, insn.operand);
                                stack.push(Integer.class);
                                break;
                            }
                            default: {
                                p.println("//Unknown instruction, ignored: ");
                            }
                        }
                        break;
                    }
                    case AbstractInsnNode.METHOD_INSN: {
                        MethodInsnNode insn = (MethodInsnNode) instruction;
                        switch (opcode) {
                            case Opcodes.INVOKESTATIC: {
                                MethodDescriptor desc = new MethodDescriptor(insn.desc);
                                int nArgs = desc.args.length;
                                String pre = mangle(insn.owner, insn.name) + "(";
                                if (desc.returnType != Void.class) {
                                    pre = String.format("s%s%d = %s", MethodDescriptor.getSafeName(desc.returnType), stackHeight - nArgs, pre);
                                }
                                StringJoiner j = new StringJoiner(", ", pre, ");\n");
                                int i = 0;
                                for (Object arg : desc.args) {
                                    i++;
                                    j.add(String.format("s%s%d", MethodDescriptor.getSafeName(arg), stackHeight - i));
                                    stack.pop();
                                }
                                if (desc.returnType != Void.class) {
                                    stack.push(desc.returnType);
                                }
                                requiredFiles.add(fixPackagePath(insn.owner));
                                p.print(j);
                                break;
                            }
                            default: {
                                p.println("//Unknown instruction, ignored: ");
                            }
                        }
                        break;
                    }
                    case AbstractInsnNode.FIELD_INSN: {
                        FieldInsnNode insn = (FieldInsnNode) instruction;
                        Object desc = MethodDescriptor.decodeDescriptor(insn.desc, new int[]{0});
                        String safeName = MethodDescriptor.getSafeName(desc);
                        String mangled = mangle(insn.owner, insn.name);
                        requiredFiles.add(fixPackagePath(insn.owner));
                        switch (opcode) {
                            case Opcodes.GETSTATIC: {
                                p.printf("s%s%d = %s;\n", safeName, stackHeight, mangled);
                                stack.push(desc);
                                break;
                            }
                            case Opcodes.PUTSTATIC: {
                                p.printf("%s = s%s%d;\n", mangled, safeName, stackHeight - 1);
                                stack.pop();
                                break;
                            }
                            default: {
                                p.println("//Unknown instruction, ignored: ");
                            }
                        }
                        break;
                    }
                    case AbstractInsnNode.VAR_INSN: {
                        VarInsnNode insn = (VarInsnNode) instruction;
                        switch (opcode) {
                            case Opcodes.ILOAD: {
                                p.printf("sI%d = aI%d;\n", stackHeight, insn.var);
                                stack.push(Integer.class);
                                break;
                            }
                            case Opcodes.ISTORE: {
                                p.printf("aI%d = sI%d;\n", insn.var, stackHeight - 1);
                                localTypes[insn.var] = Integer.class;
                                stack.pop();
                                break;
                            }
                            case Opcodes.ALOAD: {
                                p.printf("sL%d = aL%d;\n", stackHeight, insn.var);
                                stack.push(localTypes[insn.var]);
                                break;
                            }
                            case Opcodes.ASTORE: {
                                p.printf("aL%d = sL%d;\n", insn.var, stackHeight - 1);
                                localTypes[insn.var] = stack.pop();
                                break;
                            }
                            default: {
                                p.println("//Unknown instruction, ignored: ");
                            }
                        }
                        break;
                    }
                    case AbstractInsnNode.JUMP_INSN: {
                        JumpInsnNode jmpInsn = (JumpInsnNode) instruction;
                        CharSequence jmpStatement = "goto " + labelIds.get(jmpInsn.label.getLabel()) + ";\n";
                        switch (opcode) {
                            case Opcodes.GOTO: {
                                p.print(jmpStatement);
                                break;
                            }
                            case Opcodes.IF_ICMPEQ: {
                                p.printf("if(sI%d == sI%d){\n", stackHeight - 2, stackHeight - 1);
                                p.print(jmpStatement);
                                p.println("}");
                                stack.pop();
                                stack.pop();
                                break;
                            }
                            case Opcodes.IF_ICMPNE: {
                                p.printf("if(sI%d != sI%d){\n", stackHeight - 2, stackHeight - 1);
                                p.print(jmpStatement);
                                p.println("}");
                                stack.pop();
                                stack.pop();
                                break;
                            }
                            case Opcodes.IF_ICMPGT: {
                                p.printf("if(sI%d > sI%d){\n", stackHeight - 2, stackHeight - 1);
                                p.print(jmpStatement);
                                p.println("}");
                                stack.pop();
                                stack.pop();
                                break;
                            }
                            case Opcodes.IF_ICMPGE: {
                                p.printf("if(sI%d >= sI%d){\n", stackHeight - 2, stackHeight - 1);
                                p.print(jmpStatement);
                                p.println("}");
                                stack.pop();
                                stack.pop();
                                break;
                            }
                            case Opcodes.IF_ICMPLT: {
                                p.printf("if(sI%d < sI%d){\n", stackHeight - 2, stackHeight - 1);
                                p.print(jmpStatement);
                                p.println("}");
                                stack.pop();
                                stack.pop();
                                break;
                            }
                            case Opcodes.IF_ICMPLE: {
                                p.printf("if(sI%d <= sI%d){\n", stackHeight - 2, stackHeight - 1);
                                p.print(jmpStatement);
                                p.println("}");
                                stack.pop();
                                stack.pop();
                                break;
                            }
                            case Opcodes.IFEQ: {
                                p.printf("if(sI%d == 0){\n", stackHeight - 1);
                                p.print(jmpStatement);
                                p.print("}");
                                stack.pop();
                                break;
                            }
                            case Opcodes.IFNE: {
                                p.printf("if(sI%d != 0){\n", stackHeight - 1);
                                p.print(jmpStatement);
                                p.print(")");
                                stack.pop();
                                break;
                            }
                            case Opcodes.IFGT: {
                                p.printf("if(sI%d > 0){\n", stackHeight - 1);
                                p.print(jmpStatement);
                                p.print("}");
                                stack.pop();
                                break;
                            }
                            case Opcodes.IFGE: {
                                p.printf("if(sI%d >= 0){\n", stackHeight - 1);
                                p.print(jmpStatement);
                                p.print("}");
                                stack.pop();
                                break;
                            }
                            case Opcodes.IFLT: {
                                p.printf("if(sI%d < 0){\n", stackHeight - 1);
                                p.print(jmpStatement);
                                p.print(")");
                                stack.pop();
                                break;
                            }
                            case Opcodes.IFLE: {
                                p.printf("if(sI%d <= 0){\n", stackHeight - 1);
                                p.print(jmpStatement);
                                p.print(")");
                                stack.pop();
                                break;
                            }

                            default: {
                                p.println("//Unknown instruction, ignored: ");
                            }
                        }
                        break;
                    }
                    case AbstractInsnNode.LABEL: {
                        p.printf("%s: ;\n", labelIds.get(((LabelNode) instruction).getLabel()));
                        break;
                    }
                    case AbstractInsnNode.IINC_INSN: {
                        IincInsnNode insn = (IincInsnNode) instruction;
                        p.printf("aI%d = aI%d + %d;", insn.var, insn.var, insn.incr);
                    }
                    case AbstractInsnNode.FRAME:
                    case AbstractInsnNode.LINE://Ignoreds
                        break;
                    default: {
                        p.println("//Unknown instruction, ignored: ");
                    }
                }
                String trim = getInsnString(instruction, labelIds);
                p.println("//SH: " + stackHeight + " |T:" + type + " |O:" + opcode + "\n//" + trim);

                p.close();

                /* Add the current instruction stringified to our buffer */
                translated[curr] = stringWriter.getBuffer();

                /* Obtain remaining edges resorting to our cfg and add every edge to unvisited set*/
                List<OpenEdge> openEdges = new ArrayList<>();
                for (Integer i : cfg.get(curr)) {
                    OpenEdge edge = new OpenEdge(stack, localTypes, i);
                    openEdges.add(edge);
                }
                open.addAll(openEdges);
            }
        } catch (Exception e) {
            //Print as much as possible before rethrowing.
            for (CharSequence charSequence : translated) {
                if (charSequence != null) {
                    pFinal.print(charSequence);
                }
            }
            pFinal.flush();
            throw new RuntimeException(e);
        }
        /* When the graph is finally fully explored print every string */
        for (CharSequence charSequence : translated) {
            if (charSequence != null) {
                pFinal.print(charSequence);
            }
        }

        {
            System.out.println("digraph " + method.name + " {");
            for (int i = 0; i < insnList.size(); i++) {
                System.out.println("insn" + i + "[label=\""
//                        + i + ":" + getInsnString(insnList.get(i), labelIds) + "\\n"
                        + ((translated[i]!=null?translated[i]:"").toString().replace("\r", "").replace("\n", "\\n"))
                        + "\"];");
            }
            analyzer.edges.forEach((s, d) -> System.out.println("insn" + s + " -> insn" + d + ";"));
            System.out.println("START -> insn0;");
            System.out.println("START[shape=\"diamond\"];");
            System.out.println("}");
        }

        pFinal.println("}");
        return requiredFiles;
    }

    private MethodDescriptor printSignature(String owner, MethodNode method, PrintWriter pFinal, HashSet<String> gennedVars) {
        // Decode methodDescriptor
        MethodDescriptor methodDescriptor = new MethodDescriptor(method.desc);
        //Print the return type of the method
        pFinal.print(MethodDescriptor.stringify(methodDescriptor.returnType));

        pFinal.print(" " + mangle(owner, method.name) + "(");


        {
            //Print the arguments
            StringJoiner joiner = new StringJoiner(", ");
            int i = 0;


            for (Object arg : methodDescriptor.args) {
                String genned = "a" + MethodDescriptor.getSafeName(arg) + i;
                gennedVars.add(genned);
                String argS = MethodDescriptor.stringify(arg) + " " + genned;
                joiner.add(argS);
                i++;
            }
            pFinal.print(joiner.toString());
        }
        pFinal.print(")");
        return methodDescriptor;
    }

    private void emitCopy(PrintWriter print, Object type, int from, int to) {
        String safeName = MethodDescriptor.getSafeName(type);
        print.printf("s%s%d = s%s%d;\n", safeName, to, safeName, from);
    }


    private static String getInsnString(AbstractInsnNode instruction, Map<Label, String> labelMap) {
        Textifier t = new PremappedTextifier(labelMap);
        TraceMethodVisitor traceMethodVisitor = new TraceMethodVisitor(t);
        instruction.accept(traceMethodVisitor);
        StringWriter writer = new StringWriter();
        t.print(new PrintWriter(writer));
        try {
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return writer.getBuffer().toString().trim();

    }

    private static String castArray(String name, Object type) {
        return cast(name, new ArrayDescriptor(type));
    }

    private static String cast(String name, Object type) {
        return "((" + cast(type) + ")" + name + ")";
    }

    private static String cast(Object type) {
        if (type instanceof ArrayDescriptor) {
            ArrayDescriptor desc = (ArrayDescriptor) type;
            return cast(desc.innertype) + "*";
        } else {
            return MethodDescriptor.stringify(type);
        }
    }

    private static class CFGAnalyser extends Analyzer<BasicValue> {

        public final Multimap<Integer, Integer> edges = HashMultimap.create();

        public CFGAnalyser() {
            super(new BasicInterpreter());
        }

        @Override
        protected void newControlFlowEdge(int insnIndex, int successorIndex) {
            edges.put(insnIndex, successorIndex);
            super.newControlFlowEdge(insnIndex, successorIndex);
        }
    }

    private static class OpenEdge {
        public final ArrayDeque<Object> stack;
        public final Object[] localTypes;
        public final int index;

        private OpenEdge(ArrayDeque<Object> stack, Object[] localTypes, int index) {
            this.stack = stack;
            this.localTypes = localTypes;
            this.index = index;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            OpenEdge openEdge = (OpenEdge) o;
            return index == openEdge.index &&
                    stack.equals(openEdge.stack) &&
                    Arrays.equals(localTypes, openEdge.localTypes);
        }

        @Override
        public int hashCode() {
            int result = Objects.hash(stack, index);
            result = 31 * result + Arrays.hashCode(localTypes);
            return result;
        }
    }

    private static class PremappedTextifier extends Textifier {
        public PremappedTextifier(Map<Label, String> labelMap) {
            super(Opcodes.ASM7);
            labelNames = new HashMap<>(labelMap);
        }
    }
}
