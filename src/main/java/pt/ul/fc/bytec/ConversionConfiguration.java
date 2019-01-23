package pt.ul.fc.bytec;

import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldNode;
import org.objectweb.asm.tree.MethodNode;

public abstract class ConversionConfiguration {
    public abstract String getModifiers(ClassNode clazz, MethodNode method);

    public String getFieldModifiers(ClassNode clazz, FieldNode method) {
        return "";
    }

    public String getExtension() {
        return ".c";
    }

    public String getExtraHeaders() {
        return "";
    }
}
