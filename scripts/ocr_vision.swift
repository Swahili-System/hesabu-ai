#!/usr/bin/env swift
import Foundation
import ImageIO
import Vision

struct OCRPage: Codable {
    let page_index: Int
    let image_path: String
    let text: String
    let text_lines: [String]
}

func usage() -> Never {
    fputs("Usage: scripts/ocr_vision.swift <image-dir> <output-jsonl>\n", stderr)
    exit(2)
}

guard CommandLine.arguments.count == 3 else {
    usage()
}

let imageDir = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
let fileManager = FileManager.default
let encoder = JSONEncoder()
encoder.outputFormatting = [.withoutEscapingSlashes]

let files = (try fileManager.contentsOfDirectory(at: imageDir, includingPropertiesForKeys: nil))
    .filter { $0.lastPathComponent.hasPrefix("page_") }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

fileManager.createFile(atPath: outputURL.path, contents: nil)
guard let output = try? FileHandle(forWritingTo: outputURL) else {
    fputs("Could not open output file: \(outputURL.path)\n", stderr)
    exit(1)
}
defer { try? output.close() }

func pageIndex(from url: URL) -> Int? {
    let name = url.deletingPathExtension().lastPathComponent
    let number = name.replacingOccurrences(of: "page_", with: "")
    return Int(number)
}

func recognizeText(in imageURL: URL) throws -> [String] {
    guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "OCR", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not load image"])
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try handler.perform([request])

    let observations = request.results ?? []
    return observations.compactMap { observation in
        observation.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines)
    }.filter { !$0.isEmpty }
}

for file in files {
    guard let index = pageIndex(from: file) else {
        continue
    }

    do {
        let lines = try recognizeText(in: file)
        let page = OCRPage(
            page_index: index,
            image_path: file.path,
            text: lines.joined(separator: "\n"),
            text_lines: lines
        )
        let data = try encoder.encode(page)
        output.write(data)
        output.write("\n".data(using: .utf8)!)
        fputs("OCR page \(index): \(lines.count) lines\n", stderr)
    } catch {
        fputs("OCR page \(index) failed: \(error.localizedDescription)\n", stderr)
    }
}
